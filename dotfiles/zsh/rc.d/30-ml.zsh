# 30-ml.zsh — GPU observability and ML workflow commands.

# oh-my-zsh's git plugin claims `gpu` for `git push upstream`. An alias with the
# same name as a function being defined makes zsh fail to parse the definition,
# which would silently kill everything below it. On a GPU box `gpu` belongs to
# the GPU; `git push upstream` is still one keystroke away as `gpush`.
if [[ ${aliases[gpu]-} == 'git push upstream' ]]; then
  unalias gpu
  alias gpush='git push upstream'
fi

# =====================================================================
#  GPU: at a glance
# =====================================================================

# Full-screen live monitors.
alias gtop='nvitop'                 # richest: per-process VRAM, tree view, kill
alias gtopc='nvitop --compact'
alias nvt='nvtop'                   # lightweight graph view
alias gwatch='watch -n1 --color gpustat --color -cpu'

# `gpu` — one-shot readable snapshot: card state on top, processes below.
gpu() {
  local q='index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,fan.speed'
  print -P "%F{cyan}%B── GPU ──────────────────────────────────────────────%b%f"
  nvidia-smi --query-gpu="$q" --format=csv,noheader,nounits | while IFS=, read -r i name util mu mt temp pw pl clk fan; do
    local pct=$(( 100 * ${mu// /} / ${mt// /} ))
    printf "  %s[%s]%s %s\n" $'\e[1m' "${i// /}" $'\e[0m' "${name# }"
    printf "    util %s%3s%%%s   vram %s%6s%s / %s MiB (%s%%)   %s°C   %sW / %sW   %s MHz   fan %s%%\n" \
      $'\e[32m' "${util// /}" $'\e[0m' $'\e[33m' "${mu// /}" $'\e[0m' "${mt// /}" "$pct" \
      "${temp// /}" "${pw// /}" "${pl// /}" "${clk// /}" "${fan// /}"
  done
  print -P "%F{cyan}%B── processes ────────────────────────────────────────%b%f"
  local procs
  procs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  if [[ -z "$procs" ]]; then
    print -P "  %F{green}idle — no compute processes%f"
  else
    printf "  %-8s %-9s %-10s %-6s %s\n" PID VRAM USER CPU% COMMAND
    echo "$procs" | while IFS=, read -r pid mem; do
      pid=${pid// /}; mem=${mem// /}
      local user cpu cmd
      # `command ps`: bare `ps` is aliased to procs, which has different flags.
      user=$(command ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      cpu=$(command ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ')
      cmd=$(command ps -o args= -p "$pid" 2>/dev/null | cut -c1-70)
      printf "  %-8s %-9s %-10s %-6s %s\n" "$pid" "${mem}MiB" "${user:-?}" "${cpu:-?}" "${cmd:-?}"
    done
  fi
}

# `gmem` — just the VRAM number, for scripting and quick checks.
gmem() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; }

# `gwho` — who is holding the GPU, one line per process.
gwho() { nvitop --once 2>/dev/null || gpu; }

# `gkill` — pick GPU processes with fzf and terminate them. Nothing dies without
# an explicit selection, and SIGTERM is sent first so training can checkpoint.
gkill() {
  local sel
  sel=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=, read -r pid mem; do
        pid=${pid// /}
        printf "%s\t%sMiB\t%s\n" "$pid" "${mem// /}" "$(command ps -o args= -p $pid 2>/dev/null | cut -c1-90)"
      done \
    | fzf --multi --header='select GPU processes to terminate (TAB to mark)' --with-nth=1,2,3)
  [[ -z "$sel" ]] && { print "nothing selected"; return 0 }
  echo "$sel" | awk '{print $1}' | while read -r pid; do
    print -P "%F{yellow}SIGTERM -> $pid%f"; kill -TERM "$pid"
  done
}

# `gfree [MiB]` — block until at least N MiB of VRAM is free. Lets you chain a
# queued run behind a job that's finishing: `gfree 24000 && python train.py`
gfree() {
  local need=${1:-20000} used total free
  while :; do
    IFS=, read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    free=$(( ${total// /} - ${used// /} ))
    (( free >= need )) && { print -P "%F{green}${free} MiB free — go%f"; return 0 }
    print -n "\rwaiting for ${need} MiB … ${free} MiB free   "
    sleep 10
  done
}

# `gpersist` / power tuning helpers (need sudo; the 400W cap is applied at boot
# by systemd, these are for temporary experiments).
gpower() { sudo nvidia-smi -pl "${1:?usage: gpower <watts>}"; }
alias gclocks='nvidia-smi -q -d CLOCK'
alias gerrors='nvidia-smi -q -d ECC,PAGE_RETIREMENT'
alias gtopo='nvidia-smi topo -m'

# =====================================================================
#  Python environments
# =====================================================================

# `ca` — activate a conda env, with fzf picker when no name is given.
ca() {
  local env=$1
  if [[ -z "$env" ]]; then
    env=$(conda env list | awk '!/^#/ && NF {print $1}' | fzf --header='conda env') || return
  fi
  conda activate "$env"
}
alias cde='conda deactivate'
alias cls='conda env list'
# What is actually installed in the current env, newest pip installs first.
alias cpkg='pip list --format=columns'

# uv: fast venv + resolver. `uvinit` makes a project venv and activates it.
uvinit() { uv venv "${1:-.venv}" && source "${1:-.venv}/bin/activate"; }
alias uvi='uv pip install'
alias uvs='uv pip sync'
alias uvl='uv pip list'

alias py='python'
alias ipy='ipython'
alias pyv='python -c "import sys,torch;print(sys.version);print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda,\"avail\",torch.cuda.is_available())"'

# `vram` — what torch thinks it is holding, from the outside.
vram() {
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible to torch")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    p = torch.cuda.get_device_properties(i)
    print(f"[{i}] {p.name}  sm_{p.major}{p.minor}  "
          f"{(total-free)/2**30:.2f} / {total/2**30:.2f} GiB used")
PY
}

# =====================================================================
#  Running experiments
# =====================================================================

# `train <cmd...>` — run a training command with a timestamped log, GPU state
# captured before and after, wall-clock timing and a terminal bell on finish.
# Logs land in ./logs/ so they sit next to the code that produced them.
train() {
  [[ $# -eq 0 ]] && { print "usage: train <command...>"; return 1 }
  local ts=$(date +%Y%m%d-%H%M%S)
  local log="${TRAIN_LOG_DIR:-./logs}/${ts}.log"
  mkdir -p "${log:h}"
  {
    print "# started : $(date -Is)"
    print "# host    : $(hostname)"
    print "# cwd     : $PWD"
    print "# env     : ${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV:t:-system}}"
    print "# git     : $(git rev-parse --short HEAD 2>/dev/null || print 'not a repo')"
    print "# command : $*"
    print "# gpu     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    print "# ---"
  } | tee "$log"

  local start=$SECONDS
  "$@" 2>&1 | tee -a "$log"
  local rc=${pipestatus[1]} elapsed=$(( SECONDS - start ))

  {
    print "# ---"
    printf "# finished: %s  rc=%d  elapsed=%02d:%02d:%02d\n" \
      "$(date -Is)" "$rc" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
  } | tee -a "$log"

  print -n '\a'   # bell — audible finish when you've tabbed away
  if (( rc == 0 )); then
    print -P "%F{green}✓ done in ${elapsed}s — log: $log%f"
  else
    print -P "%F{red}✗ failed (rc=$rc) after ${elapsed}s — log: $log%f"
  fi
  return $rc
}

# `bigbuild <cmd...>` — temporarily lift the MAX_JOBS cap for builds that are
# not nvcc-bound. CUDA extension builds should stay at the default.
bigbuild() { MAX_JOBS=$(nproc) "$@"; }

# `retry <n> <cmd...>` — for flaky Hub downloads and preemptible steps.
retry() {
  local n=${1:?usage: retry <n> <cmd...>}; shift
  local i
  for i in {1..$n}; do
    "$@" && return 0
    print -P "%F{yellow}attempt $i/$n failed; retrying in $((i*5))s%f"
    sleep $((i*5))
  done
  return 1
}

# `ckpt [dir]` — checkpoints newest first with sizes; the usual "which run was
# that and how much disk is it eating" question.
ckpt() {
  fd -e pt -e pth -e ckpt -e safetensors -e bin . "${1:-.}" \
     -x command ls -lh --time-style=long-iso {} \
    | sort -k6,7 -r | awk '{printf "%-10s %s %s  %s\n", $5, $6, $7, $9}'
}

# `tb [logdir]` — TensorBoard bound to all interfaces so it's reachable over
# Tailscale, not just localhost.
tb() { tensorboard --logdir "${1:-./runs}" --host 0.0.0.0 --port "${2:-6006}"; }

alias jl='jupyter lab --no-browser --ip=0.0.0.0'
alias jn='jupyter notebook --no-browser --ip=0.0.0.0'

# Datasets and disk: the two things that fill this box up.
alias hfcache='command du -sh $HF_HOME/hub/* 2>/dev/null | sort -h'
alias diskhog='dust -r -n 25'

# =====================================================================
#  Prompt GPU segment (powerlevel10k custom segment)
# =====================================================================

typeset -g __GPU_STAT_CACHE="${XDG_RUNTIME_DIR:-/tmp}/gpu-prompt.stat"
typeset -g __GPU_STAT_BEAT="${XDG_RUNTIME_DIR:-/tmp}/gpu-prompt.beat"

# Touch the heartbeat and (re)start the sampler if it isn't running. Called from
# precmd, so the daemon lives exactly as long as an interactive shell is around.
__gpu_stat_heartbeat() {
  : >| $__GPU_STAT_BEAT
  [[ -f $__GPU_STAT_CACHE ]] && return
  ( $HOME/.config/zsh/bin/gpu-statd &>/dev/null & ) 2>/dev/null
}
autoload -Uz add-zsh-hook
add-zsh-hook precmd __gpu_stat_heartbeat

# p10k calls this to render the segment. Reads the cache file only — no forks,
# no nvidia-smi, so it costs nothing per prompt.
prompt_gpu() {
  [[ -r $__GPU_STAT_CACHE ]] || return
  local -a lines; lines=("${(f)$(<$__GPU_STAT_CACHE)}")
  (( ${#lines} )) || return

  local util mu mt temp pw
  local max_util=0 sum_mu=0 sum_mt=0 max_temp=0 sum_pw=0
  local l
  for l in $lines; do
    read -r util mu mt temp pw <<< "$l"
    (( util > max_util )) && max_util=$util
    (( temp > max_temp )) && max_temp=$temp
    (( sum_mu += mu )); (( sum_mt += mt )); (( sum_pw += ${pw%.*} ))
  done
  (( sum_mt > 0 )) || return

  # Colour tracks whichever pressure matters: hot card, or nearly-full VRAM.
  local mem_pct=$(( 100 * sum_mu / sum_mt ))
  local color=2                                     # green: idle/healthy
  (( max_util > 5 || mem_pct > 10 )) && color=6     # cyan: working
  (( mem_pct > 80 || max_temp > 78 ))  && color=3   # yellow: pressure
  (( mem_pct > 93 || max_temp > 84 ))  && color=1   # red: about to hurt

  local gib=$(printf '%.1f' $(( sum_mu / 1024.0 )))
  local tot=$(printf '%.0f' $(( sum_mt / 1024.0 )))
  # `%%` — p10k runs the -t text through prompt expansion, so a literal percent
  # sign has to be doubled or it silently disappears.
  p10k segment -f $color -i '󰢮' -t "${max_util}%% ${gib}/${tot}G ${max_temp}°"
}

# =====================================================================
#  Convenience
# =====================================================================

alias zrc='${EDITOR} ~/.zshrc'
alias zrcd='cd ~/.config/zsh/rc.d'
alias reload='exec zsh'
alias lg='lazygit'
alias ports='ss -tulpn'
alias myip='curl -s ifconfig.me'
alias ts='tailscale status'
alias cs='claude-statusbar'
alias cstatus='claude-statusbar'
alias speedtest='speedtest-cli --secure'
