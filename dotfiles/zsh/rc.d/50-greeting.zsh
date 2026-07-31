# 50-greeting.zsh — login banner.
#
# neofetch used to run here; it costs ~300ms and mostly reports things that
# never change. This shows the state that actually varies between logins:
# what the GPU is doing, how much disk and RAM is left, and who else is on.
# Login/SSH shells only, so nested shells and `exec zsh` stay instant.

__ml_greeting() {
  local c=$'\e[36m' b=$'\e[1m' d=$'\e[2m' r=$'\e[0m' g=$'\e[32m' y=$'\e[33m'

  printf "%s%s  %s%s  ·  %s  ·  up %s%s\n" "$b" "$c" "$(hostname)" "$r$d" \
    "$(uname -r)" "$(uptime -p | sed 's/^up //')" "$r"

  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
             --format=csv,noheader,nounits 2>/dev/null \
  | while IFS=, read -r i name util mu mt temp pw; do
      i=${i// /}; util=${util// /}; mu=${mu// /}; mt=${mt// /}; temp=${temp// /}
      local pct=$(( 100 * mu / mt ))
      local col=$g; (( pct > 80 || temp > 78 )) && col=$y
      # 20-cell bar of VRAM occupancy.
      local filled=$(( pct * 20 / 100 )) bar=''
      local n; for (( n = 0; n < 20; n++ )); do
        (( n < filled )) && bar+='█' || bar+='░'
      done
      printf "  %sGPU%s%s  %s%s%s  %s%%util  %s%s%s %s/%s MiB (%s%%)  %s°C  %sW\n" \
        "$b" "$i" "$r" "$d" "${name# }" "$r" "$util" "$col" "$bar" "$r" "$mu" "$mt" "$pct" "$temp" "${pw%%.*}"
    done

  local mem disk
  # `command` prefix throughout: df/du/ps are aliased to duf/dust/procs in
  # 20-tools.zsh, and those take different flags.
  mem=$(free -g | awk '/^Mem:/{printf "%d/%d GiB", $3, $2}')
  disk=$(command df -h --output=used,size,pcent / | tail -1 | awk '{printf "%s/%s (%s)", $1, $2, $3}')
  printf "  %sram%s %s   %sdisk /%s %s   %senv%s %s\n" \
    "$d" "$r" "$mem" "$d" "$r" "$disk" "$d" "$r" "${CONDA_DEFAULT_ENV:-none}"

  # Anything already training? Worth knowing before you launch a second job.
  local nproc_gpu
  nproc_gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . )
  if (( nproc_gpu > 0 )); then
    printf "  %s⚠ %d process(es) already on the GPU — run %sgpu%s%s to see them%s\n" \
      "$y" "$nproc_gpu" "$b" "$r$y" "" "$r"
  fi
  print
}

if [[ -o login || -n $SSH_CONNECTION ]] && [[ -z $__ML_GREETED ]]; then
  export __ML_GREETED=1
  __ml_greeting
fi
