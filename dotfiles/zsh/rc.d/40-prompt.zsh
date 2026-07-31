# 40-prompt.zsh — powerlevel10k, plus the live GPU segment.
#
# Only one prompt engine may be active. oh-my-posh and starship are both present
# on this box; running either alongside p10k makes both draw the prompt and
# corrupts the line editor. p10k wins because ~/.p10k.zsh is already tuned.

source ~/.powerlevel10k/powerlevel10k.zsh-theme
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

# The greeting banner prints output before the prompt, which `verbose` warns
# about on every start. `quiet` keeps instant prompt without the nagging.
typeset -g POWERLEVEL9K_INSTANT_PROMPT=quiet

# Put the GPU segment first on the right prompt, ahead of the conda/venv
# segments — the two things worth glancing at are "which env" and "is the card
# busy", and they end up adjacent.
typeset -g POWERLEVEL9K_CUSTOM_GPU_BACKGROUND=
typeset -g POWERLEVEL9K_GPU_VISUAL_IDENTIFIER_EXPANSION='󰢮'
POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS=(gpu $POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS)

# Show the conda env name even when it's `base` — on a box with six envs,
# silence about which one is active is how you train against the wrong stack.
typeset -g POWERLEVEL9K_ANACONDA_SHOW_ON_COMMAND=
typeset -g POWERLEVEL9K_ANACONDA_CONTENT_EXPANSION='${${${${CONDA_PROMPT_MODIFIER#\(}% }%\)}:-${CONDA_DEFAULT_ENV}}'
