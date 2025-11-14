#!/bin/bash

task_ids=("sharded-livecodebench/2883" "sharded-livecodebench/2954" "sharded-HumanEval/59" "sharded-livecodebench/2888" "sharded-livecodebench/2756" "sharded-livecodebench/2847" "sharded-livecodebench/2791" "sharded-HumanEval/109" "sharded-livecodebench/2802" "sharded-HumanEval/99" "sharded-livecodebench/2844" "sharded-livecodebench/2816" "sharded-livecodebench/2792" "sharded-livecodebench/2828" "sharded-HumanEval/39" "sharded-livecodebench/2873" "sharded-livecodebench/2811" "sharded-HumanEval/17" "sharded-livecodebench/2877" "sharded-HumanEval/128" "sharded-livecodebench/2866" "sharded-livecodebench/2754" "sharded-livecodebench/2882" "sharded-livecodebench/2755" "sharded-livecodebench/2785" "sharded-livecodebench/2856" "sharded-livecodebench/2786" "sharded-HumanEval/139" "sharded-HumanEval/158" "sharded-livecodebench/2779" "sharded-HumanEval/159" "sharded-livecodebench/2857" "sharded-livecodebench/2955" "sharded-HumanEval/5" "sharded-livecodebench/2892" "sharded-HumanEval/113" "sharded-HumanEval/141" "sharded-HumanEval/76" "sharded-HumanEval/137" "sharded-HumanEval/26" "sharded-livecodebench/2728")

for task_id in "${task_ids[@]}"; do
  python train_reinforce.py --task_id "$task_id" "$@"
done
