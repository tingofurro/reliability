import os, json, time, argparse
from datetime import datetime, timedelta
from pathlib import Path
from utils_experiments import get_experiment_type

def print_colored(text, color):
    colors = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m", "blue": "\033[94m", "reset": "\033[0m"}
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def check_experiment_for_errors(exp_path, exp_name):
    logs_path = os.path.join(exp_path, "logs.jsonl")
    
    if not os.path.exists(logs_path):
        return None
    
    errors = {"oom": [], "timeout": [], "other": []}
    with open(logs_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                log_entry = json.loads(line)
                error_type = log_entry.get("backprop_error_type")
                if error_type == "OOM":
                    errors["oom"].append({"iteration": log_entry.get("iteration", "unknown"), "error": log_entry.get("backprop_error", "Unknown error"), "line": line_num})
                elif error_type == "timeout":
                    errors["timeout"].append({"iteration": log_entry.get("iteration", "unknown"), "error": log_entry.get("backprop_error", "Unknown error"), "line": line_num})
                elif error_type and error_type != "OOM" and error_type != "timeout":
                    errors["other"].append({"iteration": log_entry.get("iteration", "unknown"), "error": log_entry.get("backprop_error", "Unknown error"), "line": line_num, "type": error_type})
            except json.JSONDecodeError:
                continue
    
    if errors["oom"] or errors["timeout"] or errors["other"]:
        return errors
    return None

def get_experiment_info(exp_path):
    args_path = os.path.join(exp_path, "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            return json.load(f)
    return {}

def get_last_modified_time(exp_path):
    logs_path = os.path.join(exp_path, "logs.jsonl")
    if os.path.exists(logs_path):
        return datetime.fromtimestamp(os.path.getmtime(logs_path))
    return None

def monitor_experiments(experiments_dir="experiments", time_window_minutes=30, continuous=False, interval_seconds=60):
    cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
    
    if not os.path.exists(experiments_dir):
        print_colored(f"Experiments directory '{experiments_dir}' not found", "red")
        return
    
    experiments = sorted([exp for exp in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, exp))])
    
    if not experiments:
        print_colored(f"No experiments found in '{experiments_dir}'", "yellow")
        return
    
    active_experiments = []
    for exp in experiments:
        exp_path = os.path.join(experiments_dir, exp)
        last_modified = get_last_modified_time(exp_path)
        
        if last_modified and last_modified >= cutoff_time:
            active_experiments.append((exp, exp_path, last_modified))
    
    if not active_experiments:
        print_colored(f"No experiments active in the last {time_window_minutes} minutes", "green")
        return
    
    print_colored(f"\nChecking {len(active_experiments)} active experiments (last {time_window_minutes} minutes):", "blue")
    print("-" * 80)
    
    errors_found = False
    for exp_name, exp_path, last_modified in active_experiments:
        errors = check_experiment_for_errors(exp_path, exp_name)
        exp_info = get_experiment_info(exp_path)
        experiment_type = get_experiment_type(exp_info)
        
        if errors:
            errors_found = True
            
            error_types = []
            if errors["oom"]:
                error_types.append("OOM")
            if errors["timeout"]:
                error_types.append("TIMEOUT")
            if errors["other"]:
                error_types.append("ERROR")
            
            error_label = " + ".join(error_types)
            print_colored(f"\n⚠️  {error_label} DETECTED in: {exp_name}", "red")
            print_colored(f"   Last modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}", "yellow")
            
            task_id = exp_info.get("task_id", "unknown")
            batch_size = exp_info.get("batch_size", "unknown")
            
            print(f"   Task: {task_id}")
            print(f"   Type: {experiment_type}, Batch size: {batch_size}")
            
            if errors["oom"]:
                print_colored(f"   OOM Errors ({len(errors['oom'])}):", "red")
                for issue in errors["oom"]:
                    print_colored(f"   - Iteration {issue['iteration']}: {issue['error'][:100]}...", "yellow")
            
            if errors["timeout"]:
                print_colored(f"   Timeout Errors ({len(errors['timeout'])}):", "red")
                for issue in errors["timeout"]:
                    print_colored(f"   - Iteration {issue['iteration']}: {issue['error'][:100]}...", "yellow")
            
            if errors["other"]:
                print_colored(f"   Other Errors ({len(errors['other'])}):", "red")
                for issue in errors["other"]:
                    print_colored(f"   - Iteration {issue['iteration']} [{issue['type']}]: {issue['error'][:100]}...", "yellow")
            
            suggestions = []
            if errors["oom"]:
                suggestions.append(f"Try reducing --batch_size (current: {batch_size})")
            if errors["timeout"]:
                suggestions.append("Try reducing --group_size or --tree_depth to process fewer samples")
            
            if suggestions:
                for suggestion in suggestions:
                    print_colored(f"   💡 Suggestion: {suggestion}", "blue")
        else:
            print_colored(f"✓ {exp_name} - {experiment_type} - No errors (last modified: {last_modified.strftime('%H:%M:%S')})", "green")
    
    print("-" * 80)
    
    if errors_found:
        print_colored(f"\n⚠️  Errors detected! Check experiments above.", "red")
    else:
        print_colored(f"\n✓ All active experiments running without errors", "green")
    
    return errors_found

def main():
    parser = argparse.ArgumentParser(description="Monitor experiments for OOM errors")
    parser.add_argument("--experiments_dir", type=str, default="experiments", help="Path to experiments directory")
    parser.add_argument("--time_window", type=int, default=30, help="Check experiments active in the last N minutes")
    parser.add_argument("--continuous", action="store_true", help="Run continuously in monitoring mode")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds (for continuous mode)")
    
    args = parser.parse_args()
    
    if args.continuous:
        print_colored(f"Starting continuous monitoring (checking every {args.interval}s, Ctrl+C to stop)", "blue")
        try:
            while True:
                monitor_experiments(args.experiments_dir, args.time_window, continuous=True, interval_seconds=args.interval)
                print(f"\nNext check in {args.interval} seconds...\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print_colored("\n\nMonitoring stopped by user", "yellow")
    else:
        monitor_experiments(args.experiments_dir, args.time_window)

if __name__ == "__main__":
    main()

