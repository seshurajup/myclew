# 20-tools.zsh — integrations for the modern CLI stack.

# ------------------------------------------------------------------ fzf
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git --exclude node_modules'
export FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"
export FZF_ALT_C_COMMAND='fd --type d --hidden --follow --exclude .git'
export FZF_DEFAULT_OPTS='
  --height 45% --layout=reverse --border=rounded --info=inline
  --marker="* " --pointer="=>" --prompt="  "
  --color=fg:-1,bg:-1,hl:#7dcfff,fg+:#c0caf5,bg+:#292e42,hl+:#7dcfff
  --color=info:#7aa2f7,prompt:#7dcfff,pointer:#bb9af7
  --color=marker:#9ece6a,spinner:#9ece6a,header:#565f89,border:#3b4261
  --bind=ctrl-/:toggle-preview,ctrl-a:select-all,ctrl-y:accept'
export FZF_CTRL_T_OPTS="--preview 'bat --style=numbers --color=always --line-range=:300 {} 2>/dev/null || eza --tree --level=2 --color=always {}'"
export FZF_ALT_C_OPTS="--preview 'eza --tree --level=2 --color=always --icons {}'"
export FZF_CTRL_R_OPTS="--preview 'echo {}' --preview-window=down:3:wrap"
source <(fzf --zsh)

# ---------------------------------------------------------------- atuin
# Shell history in SQLite: full-text search, per-directory and per-host context,
# exit codes and durations. Bound to ctrl-r only — plain up-arrow stays as zsh's
# own prefix search, which is what muscle memory expects.
eval "$(atuin init zsh --disable-up-arrow)"

# --------------------------------------------------------------- zoxide
eval "$(zoxide init zsh)"     # `z <frecent dir>`, `zi` for interactive pick

# ----------------------------------------------------------------- bat
export BAT_THEME="Coldark-Dark"
export BAT_STYLE="numbers,changes,header"
export MANPAGER="sh -c 'col -bx | bat -l man -p'"
export MANROFFOPT="-c"

# ----------------------------------------------------------------- eza
export EZA_COLORS="da=1;34:gm=1;34"
alias ls='eza --icons --group-directories-first'
alias ll='eza -l --icons --group-directories-first --git --time-style=long-iso'
alias la='eza -la --icons --group-directories-first --git --time-style=long-iso'
alias lt='eza --tree --level=2 --icons --group-directories-first'
alias ltt='eza --tree --level=3 --icons --group-directories-first'
# Biggest files here — for hunting down stray checkpoints.
alias lsize='eza -l --icons --sort=size --reverse --total-size'

# ------------------------------------------------------- misc replacements
alias cat='bat --paging=never'
alias catp='bat'
alias df='duf'
alias du='dust'
alias ps='procs'
alias top='btop'
alias help='tldr'

# --------------------------------------------------------------- direnv
# Not installed by default; guard so this file stays portable.
(( $+commands[direnv] )) && eval "$(direnv hook zsh)"
