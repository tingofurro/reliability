import random, sys, os

# these were good for Phi-4
task_ids = ["sharded-livecodebench/2756", "sharded-livecodebench/2755", "sharded-livecodebench/2847", "sharded-livecodebench/2786", "sharded-livecodebench/2791", "sharded-livecodebench/2856", "sharded-livecodebench/2857", "sharded-livecodebench/2866", "sharded-livecodebench/2882", "sharded-livecodebench/2883"]

# these are good for Qwen3-14b
task_ids = ["sharded-livecodebench/2727", "sharded-livecodebench/2754", "sharded-livecodebench/2756", "sharded-livecodebench/2792", "sharded-livecodebench/2812", "sharded-livecodebench/2828", "sharded-livecodebench/2844", "sharded-livecodebench/2845", "sharded-livecodebench/2855", "sharded-livecodebench/2856"]

learning_rates = [5e-3]

params = [
    # {"backprop_method": "kto", "sample_strategy": "tree", "tree_depth": 13, "tree_degree": 2},
    # {"backprop_method": "grpo", "sample_strategy": "iid", "group_size": 500},
    # {"backprop_method": "kto", "sample_strategy": "iid", "group_size": 500},

    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 16},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 32},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 64},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 128},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 256},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 512},

    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 128, "learning_rate": 1e-4},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 128, "learning_rate": 5e-4},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 128, "learning_rate": 1e-3},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 128, "learning_rate": 5e-3},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 128, "learning_rate": 1e-2},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 128, "learning_rate": 5e-2},
    # {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 128, "learning_rate": 1e-1},

    # {"backprop_method": "kto", "sample_strategy": "iid", "group_size": 512, "learning_rate": 5e-4},
    # {"backprop_method": "kto", "sample_strategy": "iid", "group_size": 512, "learning_rate": 1e-3},
    # {"backprop_method": "kto", "sample_strategy": "iid", "group_size": 512, "learning_rate": 5e-3},
    # {"backprop_method": "kto", "sample_strategy": "iid", "group_size": 512, "learning_rate": 1e-2},
    # {"backprop_method": "kto", "sample_strategy": "iid", "group_size": 512, "learning_rate": 2e-2},
    # {"backprop_method": "kto", "sample_strategy": "iid", "group_size": 512, "learning_rate": 5e-2},

    {"backprop_method": "grpo", "sample_strategy": "iid", "group_size": 512, "learning_rate": 1e-2},
    {"backprop_method": "rej", "sample_strategy": "iid", "group_size": 512, "learning_rate": 1e-2},
    {"backprop_method": "kto", "sample_strategy": "iid", "group_size": 512, "learning_rate": 1e-2},

]

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
