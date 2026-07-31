# 00-env.zsh — paths, editors, CUDA/ML toolchain environment.

# ---------------------------------------------------------------- path
typeset -U path PATH                      # dedupe, keep first occurrence
path=(
  $HOME/.local/bin
  $HOME/.cargo/bin
  /home/linuxbrew/.linuxbrew/bin
  /home/linuxbrew/.linuxbrew/sbin
  /usr/local/cuda/bin
  $path
)
export PATH

eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

export EDITOR=vim VISUAL=vim
export PAGER=less LESS='-R -F -X -i'

# ---------------------------------------------------------------- cuda
# /usr/local/cuda is the symlink; follow it so a toolkit upgrade needs no edit.
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# RTX 5090 = Blackwell GB202 = compute capability 12.0 (sm_120). Setting this
# stops torch/CUDA extension builds from compiling every arch under the sun.
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Building heavy CUDA libs (flash-attn, TransformerEngine, xformers) with
# MAX_JOBS=$(nproc) OOMs this box — each nvcc job eats several GB. Keep it low;
# use `bigbuild <cmd>` to temporarily raise it for pure-C++ work.
export MAX_JOBS=4

# ------------------------------------------------------------ ml runtime
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=1        # fast parallel Hub downloads
export TOKENIZERS_PARALLELISM=false       # silences the fork warning spam
export OMP_NUM_THREADS=8                  # sane default; dataloaders oversubscribe
export PYTHONBREAKPOINT=IPython.core.debugger.set_trace
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # cuts fragmentation OOMs

# Keep model/dataset caches off the root filesystem's page cache churn.
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.cache/triton}
