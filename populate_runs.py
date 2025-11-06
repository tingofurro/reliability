from evalserv_client import EvaluationServiceClient
from llms.genserv.client import GenerationServiceClient
from utils_logs import get_log_counts
import argparse, json, torch

parser = argparse.ArgumentParser()

# Basics
parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json")
# use the + to allow multiple model names
parser.add_argument("--model_name", type=str, default="gs-microsoft/phi-4")
parser.add_argument("--degree", type=int, default=100)
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
parser.add_argument("--max_concurrent_jobs_per_worker", type=int, default=10)

logs_path = f"logs/logs.jsonl"

args = parser.parse_args()

assistant_gen_client = GenerationServiceClient(base_url=f"http://localhost:5000")
eval_client = EvaluationServiceClient(base_url=f"http://localhost:5001")

# if any of the models start with gs-, then there should be only one model name
if args.model_name.startswith("gs-"):
    assistant_gen_client.load_model(args.model_name[3:], num_gpus=args.num_gpus, workers_per_gpu=1, max_concurrent_jobs_per_worker=args.max_concurrent_jobs_per_worker)
    assistant_gen_client.wait_for_service_ready()

with open(args.dataset_fn, "r") as f:
    data = json.load(f)

data = [d for d in data if d["task"] == "code"]

run_counts = get_log_counts(args.results_path)

todo_samples = []
for sample in data:
    print(sample.keys())
    key = (sample["sample_id"])
    if key not in run_counts:
        todo_samples.append(sample)

random.shuffle(todo_samples)
