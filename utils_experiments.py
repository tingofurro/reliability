from datetime import datetime
import os, sys, shutil
import subprocess


def make_exp_folder(prefix="exp", suffix=None):
    server_name = os.environ.get("SERVER_NAME", "")
    if not server_name:
        print("\033[93mSERVER_NAME environment variable is not set. Add something like 'export SERVER_NAME=A' to your .bashrc\033[0m") # needed to differentiate experiments across servers.
        server_name = ""

    exp_taken, run_idx = True, 1
    while exp_taken:
        exp_id = f"{prefix}{datetime.now().strftime('%m%d')}_{server_name}"
        if suffix:
            exp_id += f"_{suffix}"

        exp_id += f"_{str(run_idx)}"
        
        exp_folder = os.path.join(os.path.dirname(__file__), "experiments", exp_id)
        if not os.path.exists(exp_folder):
            exp_taken = False
        run_idx += 1

    os.makedirs(exp_folder, exist_ok=True)
    return exp_folder

def get_experiment_type(exp_args):
    sample_strategy = exp_args.get("sample_strategy", "iid")
    backprop_method = exp_args.get("backprop_method", "grpo")
    tree_degree = exp_args.get("tree_degree", None)
    tree_depth = exp_args.get("tree_depth", None)
    group_size = exp_args.get("group_size", None)
    
    if sample_strategy == "tree":
        return f"tree-{tree_degree}^{tree_depth}_{backprop_method}"
    else:
        return f"iid-{group_size}_{backprop_method}"
