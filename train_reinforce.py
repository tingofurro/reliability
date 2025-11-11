import argparse, json, random, torch, time, numpy as np
from utils_tmux import start_gen_and_eval_sessions
from evalserv_client import EvaluationServiceClient
from genserv_client import GenerationServiceClient
from backprop_worker import BackpropWorker
from utils import print_colored
from collections import Counter
from tasks import get_task

parser = argparse.ArgumentParser()

# Basics
parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json")
parser.add_argument("--base_model", type=str, default="microsoft/phi-4")
parser.add_argument("--degree", type=int, default=100)
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())

# Backprop
parser.add_argument("--advantage_estimation", type=str, default="zero_mean", choices=["zero_mean", "zero_mean_noneg"])
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--model_save_path", type=str, default="checkpoints/reliability_model")

args = parser.parse_args()

start_gen_and_eval_sessions()

assistant_gen_client = GenerationServiceClient(base_url=f"http://localhost:5000")
eval_client = EvaluationServiceClient(base_url=f"http://localhost:5001")

backprop_worker = BackpropWorker()

with open(args.dataset_fn, "r") as f:
    data = json.load(f)

data = [d for d in data if d["task"] == "code"]
sample = random.choice(data)

task = get_task(sample["task"])

def generate_responses(conversation, degree):

    active_jobs, active_eval_jobs = [], []
    for i in range(degree):
        job_result = assistant_gen_client.schedule_job(conversation, n_responses=1)
        active_jobs.append({"job_id": job_result["job_id"], "degree": degree, "response_index": i, "total_responses": degree})

    responses = []
    eval_job_id2response = {}

    while active_jobs or active_eval_jobs:
        # print(len(active_jobs), len(active_eval_jobs))
        for job_info in active_jobs:
            job_result = assistant_gen_client.check_job(job_info["job_id"])
            if job_result["status"] == "completed":
                response = job_result["responses"][0]
                active_jobs.remove(job_info)
                this_conversation = conversation + [{"role": "assistant", "content": response["response_text"]}]
                eval_job_result = eval_client.schedule_evaluation(conversation=this_conversation, task_name=sample["task"], sample=sample)
                active_eval_jobs.append({"job_id": eval_job_result["job_id"]})
                eval_job_id2response[eval_job_result["job_id"]] = response
                responses.append(response)

        status_counts = Counter()
        for job_info in active_eval_jobs:
            job_result = eval_client.check_job(job_info["job_id"])
            status_counts[job_result["status"]] += 1
            if job_result["status"] == "completed":
                active_eval_jobs.remove(job_info)
                response = eval_job_id2response[job_info["job_id"]]
                response["score"] = job_result["result"]["evaluation_return"]["score"]
            elif job_result["status"] == "error":
                active_eval_jobs.remove(job_info)
                response = eval_job_id2response[job_info["job_id"]]
                response["score"] = 0
        time.sleep(0.1)
    return responses

system_message = task.generate_system_prompt(sample)
input_prompt = task.populate_fully_specific_prompt(sample)

conversation = [{"role": "system", "content": system_message}, {"role": "user", "content": input_prompt}]

CURRENT_LATEST_MODEL_PATH = args.base_model
iteration = 0

while True:
    # Step 1: Forward
    # Step 1a: Load the model on vllm backend
    load_result = assistant_gen_client.load_model(CURRENT_LATEST_MODEL_PATH, num_gpus=args.num_gpus) # Be careful, this shouldn't be commented by default
    # print(f"Model load result: {load_result}")

    # Step 1b: Generate responses
    responses = generate_responses(conversation, args.degree)

    mean_score = np.mean([response["score"] for response in responses])
    print_colored(f"Mean score: {mean_score}", "green")

    # Step 1c: Unload the model
    unload_result = assistant_gen_client.unload_model()
    # print(f"Model unload result: {unload_result}")

    # Step 2: Backprop
    MODEL_PATH = f"{args.model_save_path}"
    
    backprop_args = {"learning_rate": args.learning_rate, "advantage_estimation": args.advantage_estimation, "reduction": "sum"}
    
    print(f"\n[Train] Starting backprop with {len(responses)} responses")
    backprop_results = backprop_worker.run_backprop(model_path=CURRENT_LATEST_MODEL_PATH, save_path=MODEL_PATH, conversation=conversation, responses=responses, args_dict=backprop_args, timeout=600)
    
    if backprop_results and backprop_results["any_updates"]:
        print(f"[Train] Backprop successful! Model saved to {MODEL_PATH}")
        print(f"[Train] Timings: {backprop_results['timings']}")
        CURRENT_LATEST_MODEL_PATH = MODEL_PATH
    else:
        print(f"[Train] No backprop updates applied")
    
    iteration += 1
    print(f"\n[Train] Completed iteration {iteration}\n")