import random, sys, os

task_ids = ["sharded-livecodebench/2756", "sharded-livecodebench/2755", "sharded-livecodebench/2847", "sharded-livecodebench/2786", "sharded-livecodebench/2791", "sharded-livecodebench/2856", "sharded-livecodebench/2857", "sharded-livecodebench/2866", "sharded-livecodebench/2882", "sharded-livecodebench/2883"]

learning_rates = [5e-3]
group_sizes = [10, 20, 50, 100, 200, 500]

all_experiments = []
for task_id in task_ids:
    for learning_rate in learning_rates:
        for group_size in group_sizes:
            all_experiments.append((task_id, learning_rate, group_size))

random.shuffle(all_experiments)

print(f"Running {len(all_experiments)} experiments")

for experiment in all_experiments:
    task_id, learning_rate, group_size = experiment
    print(f"Running experiment: {task_id}, {learning_rate}, {group_size}")
    os.system(f"python train_grpo.py --task_id {task_id} --learning_rate {learning_rate} --group_size {group_size}")
