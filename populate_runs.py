from llms.genserv.client import GenerationServiceClient
from evalserv_client import EvaluationServiceClient
import argparse, json, torch, random, tqdm, time
from concurrent.futures import ThreadPoolExecutor
from utils_logs import get_log_counts, log_single_run
from tasks import get_task

parser = argparse.ArgumentParser()

# Basics
parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json")
# use the + to allow multiple model names
parser.add_argument("--model_name", type=str, default="gs-microsoft/phi-4")
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
parser.add_argument("--num_runs", type=int, default=2000)
parser.add_argument("--max_concurrent_jobs_per_worker", type=int, default=11)

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

run_counts = get_log_counts(logs_path)

todo_samples = []
for sample in data:
    key = (sample["task_id"], args.model_name)
    num_todo_runs = args.num_runs - run_counts.get(key, 0)
    todo_samples += [sample] * max(num_todo_runs, 0)

random.shuffle(todo_samples)

def populate_single_run(sample):
    task = get_task(sample["task"])

    system_message = task.generate_system_prompt(sample)
    input_prompt = task.populate_fully_specific_prompt(sample)

    conversation = [{"role": "system", "content": system_message}, {"role": "user", "content": input_prompt}]

    job_schedule_result = assistant_gen_client.schedule_job(conversation, n_responses=1)
    
    job_status = {"status": "pending"}
    while job_status["status"] != "completed":
        job_status = assistant_gen_client.check_job(job_schedule_result["job_id"])
        time.sleep(0.2)
    
    response = job_status["responses"][0]

    this_conversation = conversation + [{"role": "assistant", "content": response["response_text"]}]
    eval_schedule_result = eval_client.schedule_evaluation(this_conversation, sample["task"], sample)
    eval_status = {"status": "pending"}
    while eval_status["status"] != "completed":
        eval_status = eval_client.check_job(eval_schedule_result["job_id"])
        time.sleep(0.2)
    
    eval_result = eval_status["result"]["evaluation_return"]
    
    # print(f"Sample: {sample['task_id']} | Response: {response['response_text']} | Eval Result: {eval_result}")
    log_single_run(logs_path, {"task_id": sample["task_id"], "model_name": args.model_name, "response": response["response_text"], "eval_result": eval_result})


num_workers = args.num_gpus * args.max_concurrent_jobs_per_worker

print(f">> Number of workers: {num_workers}")

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    list(tqdm.tqdm(executor.map(populate_single_run, todo_samples), total=len(todo_samples), desc="Populating runs"))
