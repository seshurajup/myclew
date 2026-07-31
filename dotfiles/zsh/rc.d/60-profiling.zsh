# 60-profiling.zsh — profiling and debugging, kept in the terminal.
#
# Layered roughly by zoom level:
#   live      — what is a running job doing right now (py-spy, dmon)
#   torch     — which ops and how much memory (torch.profiler, allocator)
#   nsight    — timeline and kernel counters (nsys, ncu)
#   correct   — is it actually right (compute-sanitizer, cuda-gdb)
#   flags     — torch/NCCL/dynamo debug environment

path=($HOME/.config/zsh/bin $path)

# =====================================================================
#  Live: what is that running job doing?
# =====================================================================

# ptrace_scope is 1 on this box: a process may only be attached to by its own
# parent. A training run started in another tmux pane is therefore off limits
# without root, so these wrap in sudo (expect a password prompt).
#
# `sudo env PATH=...` rather than plain `sudo`: sudoers sets secure_path, which
# does not include ~/.local/bin where pipx put py-spy.
#
# py-spy samples read-only and does not meaningfully slow the target.
__pyspy() {
  if [[ $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null) == 0 ]]; then
    py-spy "$@"
  else
    sudo env "PATH=$PATH" py-spy "$@"
  fi
}

# Pick a python process with fzf when no pid is given.
__pick_py_pid() {
  local line
  line=$(command ps -eo pid,etime,pcpu,rss,args --sort=-pcpu \
    | awk 'NR==1 || /python|torchrun|accelerate/' \
    | grep -v 'awk\|grep\|py-spy' \
    | fzf --header-lines=1 --header='pick a process') || return 1
  echo "$line" | awk '{print $1}'
}

# `pyspy [pid]` — live "top" of Python functions by time. The fastest way to see
# whether a training loop is in the model, the dataloader, or a host sync.
pyspy() {
  local pid=${1:-$(__pick_py_pid)} || return
  [[ -z $pid ]] && return 1
  __pyspy top --pid "$pid"
}

# `pystack [pid]` — one-shot stack dump of every thread. This is the tool for a
# training run that has stopped making progress: it shows the exact line.
pystack() {
  local pid=${1:-$(__pick_py_pid)} || return
  [[ -z $pid ]] && return 1
  __pyspy dump --pid "$pid" --locals
}

# `pyflame [pid] [seconds]` — sampled flamegraph of a live process.
pyflame() {
  local pid=${1:-$(__pick_py_pid)} || return
  [[ -z $pid ]] && return 1
  local dur=${2:-30} out="flame-${pid}-$(date +%H%M%S).svg"
  __pyspy record --pid "$pid" --duration "$dur" --output "$out" --rate 100
  print -P "%F{green}✓ $out%f"
}

# Live GPU counters, per-card and per-process. `gdmon` is the one to leave
# running in a split while a job warms up.
alias gdmon='nvidia-smi dmon -s pucvmet'   # power/util/clocks/violations/mem/enc/temp
alias gpmon='nvidia-smi pmon -s um'        # per-process sm+mem utilisation
alias gtrace='nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv -l 1'

# =====================================================================
#  torch.profiler — ops, kernels, memory
# =====================================================================

alias tprof='torch-prof'              # run a script, print op table + trace
alias tsum='torch-trace-summary'      # summarise an existing trace in terminal
alias tmem='torch-mem-prof'           # allocator history + peak/fragmentation

# `tprof-snippet` — the schedule-based profiler block to paste into a training
# loop. Profiling a whole run end to end produces a trace too big to read; this
# captures a few steady-state steps instead.
tprof-snippet() {
  bat --language=python --style=plain <<'PY'
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=5, warmup=3, active=5, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./runs/prof"),
    record_shapes=True, profile_memory=True, with_stack=True,
)
prof.start()
for step, batch in enumerate(loader):
    train_step(batch)
    prof.step()                      # must be called every iteration
    if step >= 15:
        break
prof.stop()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
PY
}

# =====================================================================
#  Nsight Systems — timeline, and its terminal statistics
# =====================================================================

# `nsysrec <cmd...>` — record a timeline. Traces the things that matter for a
# training step: CUDA, cuDNN/cuBLAS, NCCL, OS runtime (to catch blocking reads)
# and Python backtraces.
nsysrec() {
  [[ $# -eq 0 ]] && { print "usage: nsysrec <command...>"; return 1 }
  local out="nsys-$(date +%Y%m%d-%H%M%S)"
  nsys profile \
    --trace=cuda,cudnn,cublas,nvtx,osrt \
    --python-backtrace=cuda \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="$out" "$@"
  print -P "%F{green}✓ ${out}.nsys-rep%f  — summarise with: nsysstat ${out}.nsys-rep"
}

# `nsysstat <report>` — print the summary tables in the terminal. This is the
# part people usually miss: nsys does not need its GUI to be useful.
nsysstat() {
  local rep=${1:?usage: nsysstat <file.nsys-rep> [report...]}
  shift
  # `${@:-a b c}` collapses the default into ONE word, which nsys then rejects
  # as a single unknown report name. Build the array explicitly.
  local -a reports
  if (( $# )); then
    reports=("$@")
  else
    reports=(cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum)
  fi
  local r
  for r in $reports; do
    print -P "\n%F{cyan}%B── $r ──%b%f"
    # --force-export: nsys caches a .sqlite next to the report and refuses to
    # run if that cache is older than the report, which is the normal state
    # after re-recording. Stderr is kept so real failures are visible.
    nsys stats --force-export=true --report "$r" --format table "$rep" 2>&1 \
      | grep -vE '^(WARNING|Generating|Processing|Exporting|\s*$)' | head -30
  done
}

# `nsysprof <cmd...>` — record and immediately print the summary.
nsysprof() {
  local out="nsys-$(date +%Y%m%d-%H%M%S)"
  nsys profile --trace=cuda,cudnn,cublas,nvtx,osrt --python-backtrace=cuda \
    --cuda-memory-usage=true --force-overwrite=true --output="$out" "$@" \
    && nsysstat "${out}.nsys-rep"
}

# =====================================================================
#  Nsight Compute — per-kernel hardware counters
# =====================================================================

# The driver is loaded with RmProfilingAdminOnly=1, so reading hardware counters
# needs root or the kernel module option flipped (see `ncu-perm-help`).
#
# `ncurec <cmd...>` — full counter set. This is heavy: it replays each kernel
# many times, so always restrict the launch range on a real workload.
ncurec() {
  [[ $# -eq 0 ]] && { print "usage: ncurec <command...>   (see NCU_ARGS)"; return 1 }
  local out="ncu-$(date +%Y%m%d-%H%M%S)"
  sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    ncu --set ${NCU_SET:-full} \
        --launch-skip ${NCU_SKIP:-0} --launch-count ${NCU_COUNT:-10} \
        --export "$out" --force-overwrite \
        --print-summary per-kernel "$@"
  print -P "%F{green}✓ ${out}.ncu-rep%f"
}

# `ncutop <cmd...>` — the cheap version: speed-of-light numbers only (compute vs
# memory bound, achieved occupancy) for the first few kernels. Start here.
ncutop() {
  NCU_SET=speedOfLight NCU_COUNT=${NCU_COUNT:-20} ncurec "$@"
}

prof-perm-help() {
  print -P "%F{yellow}Two kernel settings limit profiling on this box.%f"
  print ""
  print -P "%B1. perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)%b (needs <= 2)"
  print "   nsys prints 'CPU IP/backtrace sampling not supported, disabling' and"
  print "   --python-backtrace produces nothing. GPU tracing is unaffected, so"
  print "   kernel/API summaries still work — only CPU-side sampling is lost."
  print "   Temporary:  sudo sysctl kernel.perf_event_paranoid=1"
  print "   Permanent:  kernel.perf_event_paranoid=1 in /etc/sysctl.d/99-perf.conf"
  print ""
  print -P "%B2. RmProfilingAdminOnly = $(grep -o 'RmProfilingAdminOnly: .' /proc/driver/nvidia/params 2>/dev/null | tail -c2)%b (needs 0 for non-root ncu)"
  print "   ncurec/ncutop already wrap the call in sudo, so they work as-is."
  print "   To profile without sudo:"
  print "     /etc/modprobe.d/nvidia-profiling.conf"
  print "       options nvidia NVreg_RestrictProfilingToAdminUsers=0"
  print "     sudo update-initramfs -u && reboot"
  print ""
  print -P "  %F{red}Both lower a security boundary; #2 also needs a reboot%f —"
  print "  deliberate choices, not things to flip casually."
}

alias ncu-perm-help=prof-perm-help

# =====================================================================
#  Correctness
# =====================================================================

# `sanitize <cmd...>` — catch out-of-bounds and misaligned device accesses.
# Slow (10-100x), but it finds custom-kernel bugs nothing else will.
sanitize()   { compute-sanitizer --tool memcheck   --launch-timeout 120 "$@"; }
sanitize-race() { compute-sanitizer --tool racecheck "$@"; }
sanitize-init() { compute-sanitizer --tool initcheck "$@"; }
sanitize-sync() { compute-sanitizer --tool synccheck "$@"; }

# `cudadbg <cmd...>` — make CUDA errors report at their real call site. Kernel
# launches are async, so without this a stack trace points at whatever
# unrelated line happened to synchronise next.
cudadbg() {
  CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 TORCH_SHOW_CPP_STACKTRACES=1 \
  CUDA_DEVICE_ASSERT=1 "$@"
}

# Anomaly detection for NaNs/inf appearing in backward.
anomaly() { PYTORCH_ANOMALY_MODE=1 TORCH_SHOW_CPP_STACKTRACES=1 "$@"; }

# =====================================================================
#  torch.compile / dynamo / distributed debug flags
# =====================================================================

# `tlog <topics> <cmd...>` — TORCH_LOGS passthrough.
#   tlog recompiles python train.py     # why is it recompiling every step
#   tlog graph_breaks,dynamo train.py   # where does the graph break
#   tlog output_code train.py           # the generated Triton kernel
tlog() {
  local topics=${1:?usage: tlog <topics> <command...>}; shift
  TORCH_LOGS="$topics" "$@"
}
# torch itself documents the valid topics when TORCH_LOGS is unparseable; it
# raises, so strip the traceback and keep the message.
tlog-topics() {
  TORCH_LOGS=help python -c "import torch" 2>&1 \
    | sed -n '/ValueError:/,$p' | sed '1s/ValueError://'
}

# Full inductor debug dump: generated code, graphs, and a debug directory.
inductor-debug() {
  TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+inductor,output_code,graph_breaks" \
  TORCHINDUCTOR_MAX_AUTOTUNE=1 "$@"
}

# Distributed: NCCL is silent until it deadlocks, so make it talk.
ncclog()   { NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL "$@"; }
ncclhang() { TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
             TORCH_NCCL_DESYNC_DEBUG=1 NCCL_DEBUG=INFO "$@"; }

# =====================================================================
#  CPU-side profiling
# =====================================================================

alias viz='viztracer'                       # viztracer -- train.py, then vizviewer
alias flamecpu='perf record -F 99 -g --'    # perf-based, needs perf_event_paranoid<=2
alias perftop='perf top'

# `bench <cmd...>` — statistically sound wall-clock comparison with warmup.
bench() { hyperfine --warmup 3 "$@"; }

# `prof-help` — because this file has more in it than anyone remembers.
prof-help() {
  bat --language=help --style=plain <<'TXT'
LIVE (a job that is already running)
  pyspy [pid]        live top of Python functions        pystack [pid]  stacks now (hangs)
  pyflame [pid] [s]  sampled flamegraph SVG              gdmon          live GPU counters
  gpmon              per-process GPU utilisation         gtrace         csv sample every 1s

TORCH
  tprof script.py    op table + chrome trace             tsum trace.json   summarise a trace
  tmem script.py     peak/fragmentation + call sites     tprof-snippet     in-loop profiler code

NSIGHT
  nsysprof <cmd>     record timeline + print summary     nsysstat <rep>    tables from a report
  ncutop <cmd>       kernel speed-of-light (start here)  ncurec <cmd>      full counters
  prof-perm-help     why ncu needs sudo / why CPU sampling is off

CORRECTNESS
  cudadbg <cmd>      sync launches, real stack traces    anomaly <cmd>     find NaN in backward
  sanitize <cmd>     device memcheck                     sanitize-race     race detection

COMPILE / DISTRIBUTED
  tlog <topics> <cmd>   recompiles, graph_breaks, output_code   tlog-topics  list topics
  inductor-debug <cmd>  dump generated Triton code
  ncclog <cmd>          NCCL init/collective logging     ncclhang <cmd>    deadlock forensics

CPU
  bench <cmd>        hyperfine timing                    viz               viztracer
TXT
}
