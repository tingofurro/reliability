import ujson as json

def get_log_counts(logs_path):
    log_counts = {}
    with open(logs_path, "r") as f:
        for line in f:
            log = json.loads(line)
            key = (log["sample_id"], log["model_name"])
            if key not in log_counts:
                log_counts[key] = 0
            log_counts[key] += 1
    return log_counts