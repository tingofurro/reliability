import ujson as json
import os

def get_log_counts(logs_path):
    log_counts = {}
    if not os.path.exists(logs_path):
        return log_counts
    with open(logs_path, "r") as f:
        for line in f:
            log = json.loads(line)
            key = (log["task_id"], log["model_name"])
            if key not in log_counts:
                log_counts[key] = 0
            log_counts[key] += 1
    return log_counts

def log_single_run(logs_path, run_info):
    with open(logs_path, "a") as f:
        f.write(json.dumps(run_info) + "\n")
