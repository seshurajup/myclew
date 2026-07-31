# 10-options.zsh — history, completion, navigation, key bindings.

# ------------------------------------------------------------- history
HISTFILE=$HOME/.zsh_history
HISTSIZE=200000
SAVEHIST=200000
setopt EXTENDED_HISTORY          # record timestamp + duration
setopt INC_APPEND_HISTORY        # write as you go, not just on exit
setopt SHARE_HISTORY             # all live shells see each other's commands
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_IGNORE_SPACE         # leading space = don't record (secrets, keys)
setopt HIST_REDUCE_BLANKS
setopt HIST_VERIFY               # expand !! for review instead of running blind

# ---------------------------------------------------------- navigation
setopt AUTO_CD                   # `~/proj` alone cds there
setopt AUTO_PUSHD PUSHD_IGNORE_DUPS PUSHD_SILENT
DIRSTACKSIZE=20
setopt EXTENDED_GLOB             # ^, ~, #  in globs
setopt GLOB_DOTS                 # globs match dotfiles
setopt NUMERIC_GLOB_SORT         # checkpoint-2 before checkpoint-10
setopt INTERACTIVE_COMMENTS
setopt NO_BEEP
setopt LONG_LIST_JOBS

# ---------------------------------------------------------- completion
zstyle ':completion:*' matcher-list 'm:{a-zA-Z-_}={A-Za-z_-}' 'r:|=*' 'l:|=* r:|=*'
zstyle ':completion:*' menu select
zstyle ':completion:*' group-name ''
zstyle ':completion:*:descriptions' format '%F{yellow}── %d ──%f'
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path "$HOME/.cache/zsh/zcompcache"
zstyle ':completion:*:*:kill:*:processes' list-colors '=(#b) #([0-9]#)*=0=01;31'
# Don't offer files we'd never edit when completing an editor argument.
zstyle ':completion:*:*:(vim|nvim|bat|less):*' ignored-patterns '*.(pyc|pt|bin|safetensors|ckpt|npy|npz)'

# ------------------------------------------------------------ keybinds
bindkey -e                                        # emacs mode
bindkey '^[[A' history-substring-search-up   2>/dev/null
bindkey '^[[B' history-substring-search-down 2>/dev/null
bindkey '^[[1;5C' forward-word                    # ctrl-right
bindkey '^[[1;5D' backward-word                   # ctrl-left
bindkey '^[[3~'   delete-char
bindkey '^[[H'    beginning-of-line
bindkey '^[[F'    end-of-line
bindkey '^U'      backward-kill-line              # bash-like, not kill-whole-line

# Edit the current command line in $EDITOR with ctrl-x ctrl-e — invaluable for
# the long multi-line torchrun/accelerate invocations.
autoload -Uz edit-command-line
zle -N edit-command-line
bindkey '^X^E' edit-command-line
