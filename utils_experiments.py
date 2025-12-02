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
        exp_id = f"{prefix}{datetime.now().strftime('%m%d')}_{server_name}_{str(run_idx)}"
        if suffix:
            exp_id += f"_{suffix}"
        exp_folder = os.path.join(os.path.dirname(__file__), "experiments", exp_id)
        if not os.path.exists(exp_folder):
            exp_taken = False
        run_idx += 1

    os.makedirs(exp_folder, exist_ok=True)
    return exp_folder
