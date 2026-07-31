#!/usr/bin/env bash
# myclew — mirror ALL fleet agent code + competition experiment code into this repo and make a
# STABLE commit every run IF there are pending changes. "Stable" = every .py byte-compiles first;
# a syntactically-broken tree is NEVER committed. Intended to run every 3 hours via cron.
set -uo pipefail
REPO="/home/seshu/kaggle/2026/myclew"
BIOHUB="/home/seshu/kaggle/2026/biohub-cell-tracking-during-development"
ROGII="/home/seshu/kaggle/2026/rogii-wellbore-geology-prediction"
YOUTUBE="/home/seshu/kaggle/2026/youtube"
cd "$REPO" || exit 1

RSYNC_EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' --exclude 'venv'
  --exclude 'cellmot_venv' --exclude 'site-packages' --exclude '*.log' --exclude '.ipynb_checkpoints'
  --exclude 'input' --exclude 'output' --exclude 'wheels' --exclude '*.whl' --exclude '.git'
  --exclude 'mlruns' --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.csv'
  --exclude '*.parquet' --exclude '*.npy' --exclude '*.npz' --exclude 'research' --exclude 'kernels'
  --exclude 'external' --exclude 'notebooks' --exclude 'extracted.py' --exclude 'scratchpad')

# 1) the agents + their verifiers (source of truth = biohub/fleet_agents)
rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$BIOHUB/fleet_agents/"       "$REPO/fleet_agents/"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$BIOHUB/test_fleet_agents/"  "$REPO/test_fleet_agents/"

# 2) per-competition experiment code we build (no data, just code + configs)
mkdir -p "$REPO/competitions/rogii-wellbore-geology-prediction"
rsync -a "${RSYNC_EXCLUDES[@]}" --include '*/' --include '*.py' --include '*.yml' --include '*.yaml' \
  --include '*.sh' --include '*.md' --exclude '*' \
  "$ROGII/" "$REPO/competitions/rogii-wellbore-geology-prediction/"

# 2a) biohub competition code + the learning packs (CODE ONLY — input/, models/, research/ and the
# multi-GB experiment outputs stay on the box; myclew is the single GitHub home for SOURCE).
mkdir -p "$REPO/competitions/biohub-cell-tracking-during-development"
# rsync: FIRST matching rule wins, so every --exclude must precede the --include filters or files
# inside an excluded dir match '*.py' first and get copied anyway.
rsync -a --delete "${RSYNC_EXCLUDES[@]}" \
  --exclude 'tools/' --exclude 'learning/' --exclude 'public_notebooks/' \
  --exclude 'docs/public_nb_lineage/' --exclude 'scratchpad/' --exclude 'scratch_ksubmit/' \
  --exclude 'docs/gm_writeups/_github/' --exclude '*_notebook_raw.py' --exclude '*notebook_code.py' \
  --exclude 'config/_auto/' --exclude '*token*.json' --exclude '*secret*' --exclude '*credential*' \
  --exclude 'kaggle.json' --exclude '.env' --exclude '*.pem' --exclude '*.key' \
  --exclude 'fleet_agents/' --exclude 'test_fleet_agents/' --exclude 'models/' \
  --exclude 'experiments/' --exclude 'model_scratch/' --exclude 'submissions/' \
  --include '*/' --include '*.py' --include '*.yml' --include '*.yaml' \
  --include '*.sh' --include '*.md' --include '*.learning' --include '*.json' --exclude '*' \
  "$BIOHUB/" "$REPO/competitions/biohub-cell-tracking-during-development/"

# 2a2) the LEARNING library at the repo root, NOT under a competition. These are paper packs and
# lesson files (.learning) that the :7777 hub serves across every competition — they are a library, not
# biohub content. public_pull/ is downloaded Kaggle notebooks (third-party, does not compile) and stays out.
mkdir -p "$REPO/learning"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" \
  --exclude 'public_pull/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '*_notebook_raw.py' --exclude '*notebook_code.py' \
  --include '*/' --include '*.py' --include '*.learning' --include '*.yml' --include '*.yaml' \
  --include '*.md' --include '*.json' --exclude '*' \
  "$BIOHUB/learning/" "$REPO/learning/"

# 2b2) the shared RUNTIME: :7777 knowledge hub (researchpapers/app.py) and :7788 runboard
# (store.py, runtime_cli.py, fleet/) plus the training service. This is OUR code and serves every
# competition, so it lives at the repo root rather than under one comp. research_refs/ and .venv/ are
# third-party checkouts and stay out.
mkdir -p "$REPO/tools/researchpapers"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" \
  --exclude 'research_refs/' --exclude '.venv/' --exclude 'output/' --exclude 'logs/' \
  --exclude 'data/' --exclude '.research-mvp-data/' --exclude '*token*.json' --exclude '*secret*' \
  --include '*/' --include '*.py' --include '*.md' --include '*.yml' --include '*.yaml' \
  --include '*.html' --include '*.css' --include '*.js' --include '*.sql' --exclude '*' \
  "$BIOHUB/tools/researchpapers/" "$REPO/tools/researchpapers/"

# 2c) the YouTube curriculum — lesson sources, props and the Remotion composition code. Rendered video
# (gallery/), node_modules and the vendored checkout are regenerable/huge and never tracked here.
mkdir -p "$REPO/youtube"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" \
  --exclude 'gallery/' --exclude 'vendor/' --exclude 'node_modules/' --exclude '*.mp4' \
  --exclude '*.wav' --exclude '*.mp3' --exclude '*.m4a' --exclude '.git' \
  "$YOUTUBE/" "$REPO/youtube/"

# 2b) shell environment — the zsh/ML setup this box's work depends on.
# Explicit file list, never a whole-directory rsync: ~/.zsh_history and the atuin
# db live next to these and must never be pushed.
mkdir -p "$REPO/dotfiles/zsh/rc.d" "$REPO/dotfiles/zsh/bin"
cp -p "$HOME/.zshrc"            "$REPO/dotfiles/zsh/zshrc"
cp -p "$HOME/.zshenv"           "$REPO/dotfiles/zsh/zshenv"
cp -p "$HOME/.p10k.zsh"         "$REPO/dotfiles/zsh/p10k.zsh"
rsync -a --delete "$HOME/.config/zsh/rc.d/" "$REPO/dotfiles/zsh/rc.d/"
rsync -a --delete "$HOME/.config/zsh/bin/"  "$REPO/dotfiles/zsh/bin/"

# 3) STABLE guard — every tracked .py must byte-compile, else abort the commit
if ! python -m py_compile $(find "$REPO/fleet_agents" "$REPO/test_fleet_agents" "$REPO/competitions" -name '*.py') 2>/tmp/myclew_pycompile.err; then
  echo "[myclew] STABLE-GUARD FAILED — .py compile errors, NOT committing:"; cat /tmp/myclew_pycompile.err; exit 2
fi

# 3b) same guard for the shell config — a zshrc that fails to parse locks you
# out of a working login shell, so it must never reach a "stable" snapshot.
# `-n` only parses its first file argument, so check them one at a time.
: >/tmp/myclew_zshparse.err
ZSH_BAD=0
for f in "$REPO/dotfiles/zsh/zshrc" "$REPO"/dotfiles/zsh/rc.d/*.zsh; do
  zsh -n "$f" 2>>/tmp/myclew_zshparse.err || ZSH_BAD=1
done
# dotfiles/zsh/bin holds a mix of bash and python helpers — dispatch on shebang
# rather than assuming shell, or `bash -n` silently mis-checks the .py ones.
for f in "$REPO"/dotfiles/zsh/bin/*; do
  [ -f "$f" ] || continue
  case "$(head -1 "$f")" in
    *python*) python -m py_compile "$f" 2>>/tmp/myclew_zshparse.err || ZSH_BAD=1 ;;
    *zsh*)    zsh -n "$f"              2>>/tmp/myclew_zshparse.err || ZSH_BAD=1 ;;
    *)        bash -n "$f"             2>>/tmp/myclew_zshparse.err || ZSH_BAD=1 ;;
  esac
done
if [ "$ZSH_BAD" -ne 0 ]; then
  echo "[myclew] STABLE-GUARD FAILED — shell config parse errors, NOT committing:"
  cat /tmp/myclew_zshparse.err; exit 2
fi

# 4) commit only if there are pending changes
git add -A
if git diff --cached --quiet; then
  echo "[myclew] no pending changes — nothing to commit ($(date '+%F %T'))"; exit 0
fi
N=$(git diff --cached --name-only | wc -l)
if ! git -c user.name="SeshurajuP" -c user.email="seshurajup@gmail.com" \
     commit -q -m "stable snapshot $(date '+%F %H:%M') — ${N} files"; then
  echo "[myclew] COMMIT REJECTED (hook or guard) — nothing pushed"; exit 4
fi
git push -q origin HEAD 2>&1 | tail -2 || { echo "[myclew] push failed"; exit 3; }
echo "[myclew] committed + pushed ${N} changed files ($(date '+%F %T'))"
