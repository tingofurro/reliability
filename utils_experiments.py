from datetime import datetime
import os, sys, shutil
import subprocess


def timestamp():
    return f"\033[94m[{datetime.now().strftime('%H:%M:%S')}]\033[0m"


def get_ssh_hosts(prefix=None):
    ssh_config_path = os.path.expanduser("~/.ssh/config")
    hosts = []
    try:
        with open(ssh_config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Host ') and not line.startswith('Host *'):
                    host_value = line.split('Host ', 1)[1].strip()
                    for host in host_value.split():
                        if '*' not in host and '?' not in host and (prefix is None or host.startswith(prefix)):  # Skip wildcards
                            hosts.append(host)
    except FileNotFoundError:
        print(f"Warning: SSH config file not found at {ssh_config_path}")
        return []
    except Exception as e:
        print(f"Warning: Error reading SSH config: {e}")
        return []
    
    return hosts


def make_exp_folder(prefix="exp"):
    server_name = os.environ.get("SERVER_NAME", "")
    if not server_name:
        print("\033[93mSERVER_NAME environment variable is not set. Add something like 'export SERVER_NAME=A' to your .bashrc\033[0m") # needed to differentiate experiments across servers.
        server_name = ""

    exp_taken, run_idx = True, 1
    while exp_taken:
        exp_id = f"{prefix}{datetime.now().strftime('%m%d')}_{server_name}_{str(run_idx)}"
        exp_folder = os.path.join(os.path.dirname(__file__), "experiments", exp_id)
        if not os.path.exists(exp_folder):
            exp_taken = False
        run_idx += 1

    os.makedirs(exp_folder, exist_ok=True)
    return exp_folder


def sync_experiments(machines=None, skip_models=True, skip_trees=True, experiment_filter=None): 
    # If no machines specified, get all hosts from SSH config
    if machines is None:
        machines = get_ssh_hosts(prefix="sin") + get_ssh_hosts(prefix="pal")

    print(f"{timestamp()} Syncing experiments from {machines}...")
    # local_experiments_dir = os.path.join(os.environ["HOME"], "mtco/experiments/")
    local_experiments_dir = os.path.join(os.path.dirname(__file__), "experiments/")
    os.makedirs(local_experiments_dir, exist_ok=True)
    
    sync_summary = {}
    
    for machine in machines:
        print(f"{timestamp()} Syncing experiments from {machine}...")
        sync_summary[machine] = {"synced": [], "skipped": [], "errors": []}
        
        try:
            # Get list of experiments on remote machine
            # remote_experiments_dir = f"{machine}:~/mtco/experiments/"
            
            # Check if remote experiments directory exists
            check_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=5", machine, "test -d ~/mtco_old/experiments && ls ~/mtco_old/experiments || echo NO_EXPERIMENTS"]
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=20)
            
            if result.returncode != 0:
                sync_summary[machine]["errors"].append(f"Failed to connect to {machine}")
                print(f"{timestamp()}   Error: Failed to connect to {machine}")
                continue
                
            if result.stdout.strip() == "NO_EXPERIMENTS":
                print(f"{timestamp()}   No experiments directory found on {machine}")
                continue
                
            remote_exp_folders = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            
            if not remote_exp_folders:
                print(f"{timestamp()}   No experiments found on {machine}")
                continue
                
            for exp_folder in remote_exp_folders:
                if not exp_folder:  # Skip empty strings
                    continue
                
                # Apply experiment filter if provided
                if experiment_filter and experiment_filter not in exp_folder:
                    continue
                    
                local_exp_path = os.path.join(local_experiments_dir, exp_folder)                
                # Use rsync to sync the folder (skips files with same size)
                remote_exp_path = f"{machine}:~/mtco_old/experiments/{exp_folder}/"
                
                # Create local directory if it doesn't exist
                os.makedirs(local_exp_path, exist_ok=True)

                # Build rsync command with excludes
                excludes = ["--exclude=models/", "--exclude=models/**"] # always exclude the latest model
                if skip_models:
                    excludes.extend(["--exclude=best_model/", "--exclude=best_model/**", "--exclude=*.safetensors"])
                if skip_trees:
                    excludes.extend(["--exclude=tree_logs.jsonl"])
                
                exclude_str = " ".join(excludes)
                rsync_cmd = ["rsync", "-avz", "--size-only", "--progress"] + excludes + ["-e", "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5", remote_exp_path, local_exp_path]
                
                print(f"{timestamp()}   Syncing {exp_folder}...")
                # Run without capture_output to show progress in terminal
                rsync_result = subprocess.run(rsync_cmd, timeout=200)
                
                if rsync_result.returncode == 0:
                    sync_summary[machine]["synced"].append(os.path.basename(local_exp_path))
                    print(f"{timestamp()}   ✓ Successfully synced {exp_folder}")
                else:
                    error_msg = f"Rsync failed for {exp_folder} (exit code: {rsync_result.returncode})"
                    sync_summary[machine]["errors"].append(error_msg)
                    print(f"{timestamp()}   ✗ {error_msg}")
                    
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout while connecting to {machine}"
            sync_summary[machine]["errors"].append(error_msg)
            print(f"{timestamp()}   ✗ {error_msg}")
        except Exception as e:
            error_msg = f"Unexpected error with {machine}: {str(e)}"
            sync_summary[machine]["errors"].append(error_msg)
            print(f"{timestamp()}   ✗ {error_msg}")
    
    # Print summary
    print(f"{timestamp()} " + "\n" + "="*50)
    print(f"{timestamp()} SYNC SUMMARY")
    print("="*50)
    for machine, summary in sync_summary.items():
        print(f"\n{machine}:")
        if summary["synced"]:
            print(f"  Synced: {', '.join(summary['synced'])}")
        if summary["skipped"]:
            print(f"  Skipped: {', '.join(summary['skipped'])}")
        if summary["errors"]:
            print(f"  Errors: {len(summary['errors'])} error(s)")
            for error in summary["errors"]:
                print(f"    - {error}")
    
    return sync_summary



if __name__ == "__main__":
    import time

    # Example usage
    while True:
        sync_experiments()
        time.sleep(10*60)
    # print(get_ssh_hosts(prefix="sin"))
