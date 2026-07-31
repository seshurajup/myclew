#!/bin/bash
cd /home/seshu/kaggle/2026/biohub-cell-tracking-during-development
for cfg in base rot90 gamma contrast noise; do
  echo "=== TRAINING $cfg $(date) ===" >> config/aug_ablation/ablation.log
  bash start_train.sh config/aug_ablation/$cfg.yml > config/aug_ablation/${cfg}.log 2>&1
  # extract the best golden-12 test node-recall from the log
  rec=$(grep -oE "recall[=: ]+[0-9.]+|node_recall[=: ]+[0-9.]+" config/aug_ablation/${cfg}.log | grep -oE "[0-9.]+" | sort -rn | head -1)
  echo "$cfg  best_golden12_recall=$rec" >> config/aug_ablation/RESULTS.txt
done
echo "ABLATION DONE $(date)" >> config/aug_ablation/ablation.log
