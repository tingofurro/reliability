import random, sys, os

task_ids = ["sharded-livecodebench/2756", "sharded-livecodebench/2755", "sharded-livecodebench/2847", "sharded-livecodebench/2786", "sharded-livecodebench/2791", "sharded-livecodebench/2856", "sharded-livecodebench/2857", "sharded-livecodebench/2866", "sharded-livecodebench/2882", "sharded-livecodebench/2883"]

learning_rates = [5e-3]
# group_sizes = [10, 20, 50, 100, 200, 500]
# params = [{"tree_depth": 7, "tree_degree": 2}, {"tree_depth": 8, "tree_degree": 2}, {"tree_depth": 9, "tree_degree": 2}]
params = [{"backprop_method": "kto"}, {"backprop_method": "sft"}, {"backprop_method": "grpo"}]
all_experiments = []
for task_id in task_ids:
    for learning_rate in learning_rates:
        for param in params:
            all_experiments.append((task_id, learning_rate, param))

random.shuffle(all_experiments)

print(f"Running {len(all_experiments)} experiments")

for experiment in all_experiments:
    task_id, learning_rate, param = experiment
    print(f"Running experiment: {task_id}, {learning_rate}, {param}")

    param_str = " ".join([f"--{k} {v}" for k, v in param.items()])
    command = f"python train.py --task_id {task_id} --learning_rate {learning_rate} {param_str}"
    print(command)
    os.system(command)
