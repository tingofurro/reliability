import argparse, json, random, torch, time, numpy as np, re, os, tqdm
from utils import print_colored, DoublePrint, get_git_version
from llms.genserv.client import GenerationServiceClient
from evalserv_client import EvaluationServiceClient
from utils_tmux import start_gen_and_eval_sessions
from concurrent.futures import ThreadPoolExecutor
from utils_experiments import make_exp_folder
from backprop_worker import BackpropWorker
from collections import Counter
from tasks import get_task

def extract_answer(response):
    # extract everything between ```python and ```
    try:
        return response.split("```python")[1].split("```")[0]
    except:
        return response

parser = argparse.ArgumentParser()

# Basics
parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json")
parser.add_argument("--base_model", type=str, default="microsoft/phi-4")
parser.add_argument("--task_id", type=str, default="sharded-livecodebench/2857")
parser.add_argument("--sample_strategy", type=str, default="tree", choices=["iid", "tree"])
parser.add_argument("--group_size", type=int, default=100)
parser.add_argument("--tree_degree", type=int, default=2)
parser.add_argument("--tree_depth", type=int, default=13)

parser.add_argument("--num_eval_runs", type=int, default=500)
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())

# Backprop
parser.add_argument("--backprop_method", type=str, default="grpo", choices=["grpo", "kto", "sft"])
parser.add_argument("--kto_margin", type=float, default=3.0)
parser.add_argument("--advantage_estimation", type=str, default="zero_mean", choices=["zero_mean", "zero_mean_noneg"])
parser.add_argument("--learning_rate", type=float, default=5e-3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_iterations", type=int, default=25)

args = parser.parse_args()

suffix = f"sample_{args.task_id.replace('/', '_')}_gs{args.group_size}"

start_gen_and_eval_sessions()

exp_folder = make_exp_folder(suffix=suffix)
print(f"Experiment folder: {exp_folder}")
model_save_path = os.path.join(exp_folder, "model")
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path, exist_ok=True)

args_path = os.path.join(exp_folder, "args.json")
run_params = vars(args)
run_params["git_version"] = get_git_version()
run_params["experiment_name"] = exp_folder.split("/")[-1]
with open(args_path, "w") as f:
    json.dump(run_params, f, indent=4)

logs_path = os.path.join(exp_folder, "logs.jsonl")
unique_answer_path = os.path.join(exp_folder, "unique_answers.jsonl")
DoublePrint(os.path.join(exp_folder, "run_logs.ans"))

print(run_params)

assistant_gen_client = GenerationServiceClient(base_url=f"http://localhost:5000")
eval_client = EvaluationServiceClient(base_url=f"http://localhost:5001")

backprop_worker = BackpropWorker()

with open(args.dataset_fn, "r") as f:
    data = json.load(f)

data = [d for d in data if d["task"] == "code"]
sample = [d for d in data if d["task_id"] == args.task_id][0]

task = get_task(sample["task"])

def generate_responses(conversation, group_size):
    active_jobs, active_eval_jobs = [], []
    for i in range(group_size):
        job_result = assistant_gen_client.schedule_job(conversation, n_responses=1)
        active_jobs.append({"job_id": job_result["job_id"], "group_size": group_size, "response_index": i, "total_responses": group_size})

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
            if job_result["status"] == "completed" and "evaluation_return" in job_result["result"]:
                active_eval_jobs.remove(job_info)
                response = eval_job_id2response[job_info["job_id"]]
                response["score"] = job_result["result"]["evaluation_return"]["score"]
            elif job_result["status"] == "error" or (job_result["status"] == "completed" and "evaluation_return" not in job_result["result"]):
                active_eval_jobs.remove(job_info)
                response = eval_job_id2response[job_info["job_id"]]
                response["score"] = 0
        time.sleep(0.1)
    return responses

def run_evaluation_phase(conversation, num_eval_runs):
    responses = generate_responses(conversation, num_eval_runs)
    return responses

def generate_tree_responses(conversation, tree_depth, tree_degree):
    T1 = time.time()
    tree_job = assistant_gen_client.build_tree(conversation, depth=tree_depth, degree=tree_degree)
    job_id = tree_job["job_id"]
    
    responses = []
    eval_job_id2response = {}
    active_eval_jobs = []
    tree_complete = False
    
    while not tree_complete or active_eval_jobs:
        # Check for new tree nodes
        if not tree_complete:
            tree_status = assistant_gen_client.check_on_tree(job_id, only_new=True)
            
            if tree_status.get("status") == "completed":
                tree_complete = True
                T2 = time.time()
                print(f"Tree generation completed in {T2 - T1:.2f} seconds")
            
            # Schedule evaluations for newly generated nodes
            new_nodes = tree_status.get("tree", [])
            for response in new_nodes:
                responses.append(response)
                this_conversation = conversation + [{"role": "assistant", "content": response["response_text"]}]
                eval_job_result = eval_client.schedule_evaluation(conversation=this_conversation, task_name=sample["task"], sample=sample)
                active_eval_jobs.append({"job_id": eval_job_result["job_id"]})
                eval_job_id2response[eval_job_result["job_id"]] = response
        
        # Check on active evaluation jobs
        for job_info in active_eval_jobs[:]:
            job_result = eval_client.check_job(job_info["job_id"])
            if job_result["status"] == "completed" and "evaluation_return" in job_result["result"]:
                active_eval_jobs.remove(job_info)
                response = eval_job_id2response[job_info["job_id"]]
                response["score"] = job_result["result"]["evaluation_return"]["score"]
            elif job_result["status"] == "error" or (job_result["status"] == "completed" and "evaluation_return" not in job_result["result"]):
                active_eval_jobs.remove(job_info)
                response = eval_job_id2response[job_info["job_id"]]
                response["score"] = 0
        
        time.sleep(0.1)
    
    T3 = time.time()
    print(f"Total time (tree + eval overlapped): {T3 - T1:.2f} seconds")
    return responses

def run_training_phase(conversation, sample_strategy, group_size, tree_depth, tree_degree):
    if sample_strategy == "iid":
        responses = generate_responses(conversation, group_size)
    elif sample_strategy == "tree":
        responses = generate_tree_responses(conversation, tree_depth, tree_degree)
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
    # responses = generate_responses(conversation, args.group_size + args.num_eval_runs)
    print(">> Starting evaluation and training phases in parallel")
    with ThreadPoolExecutor(max_workers=2) as executor:
        training_future = executor.submit(run_training_phase, conversation, args.sample_strategy, args.group_size, args.tree_depth, args.tree_degree)
        evaluation_future = executor.submit(run_evaluation_phase, conversation, args.num_eval_runs)
        training_responses = training_future.result()
        evaluation_responses = evaluation_future.result()

    # responses = evaluation_responses + training_responses

    for response in training_responses + evaluation_responses:
        response["answer"] = extract_answer(response["response_text"])
        response["answer2"] = re.sub(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|#.*?$)', '', response["answer"], flags=re.DOTALL | re.MULTILINE)
        response["answer2"] = "\n".join([line for line in response["answer2"].split("\n") if line.strip()]) # remove any empty lines

    # compute the uniqueness of the answers
    unique_answers = set([response["answer2"] for response in evaluation_responses])

    unique_correct_answers = sorted(set([response["answer2"] for response in evaluation_responses if response["score"] == 1]))

    response_logprobs = [response["logprobs"] for response in evaluation_responses]
    correct_logprobs = [response["logprobs"] for response in evaluation_responses if response["score"] == 1]
    incorrect_logprobs = [response["logprobs"] for response in evaluation_responses if response["score"] != 1]
    # print("RESPONSE LOGPROBS:")
    # print(response_logprobs)

    mean_train_score = np.mean([response["score"] for response in training_responses])
    mean_eval_score = np.mean([response["score"] for response in evaluation_responses])
    uniqueness = 100.0 * len(unique_answers) / len(evaluation_responses)
    print_colored(f"Mean train score: {mean_train_score}", "green")
    print_colored(f"Mean eval score: {mean_eval_score} (Uniqueness: {len(unique_answers) / len(evaluation_responses)} ({uniqueness:.2f}))", "green")

    # Step 1c: Unload the model
    unload_result = assistant_gen_client.unload_model()
    # print(f"Model unload result: {unload_result}")

    # Step 2: Backprop
    MODEL_PATH = f"{model_save_path}"
    
    backprop_args = {"backprop_method": args.backprop_method, "learning_rate": args.learning_rate, "advantage_estimation": args.advantage_estimation, "batch_size": args.batch_size, "reduction": "sum", "kto_margin": args.kto_margin}
    
    print(f"\n[Train] Starting backprop with {len(training_responses)} responses")
    backprop_results = backprop_worker.run_backprop(model_path=CURRENT_LATEST_MODEL_PATH, save_path=MODEL_PATH, conversation=conversation, responses=training_responses, args_dict=backprop_args, timeout=1800)
    
    backprop_error = None
    backprop_error_type = None
    if backprop_results is None:
        backprop_error = "Backprop worker timed out after 1800 seconds"
        backprop_error_type = "timeout"
        print_colored(f"[Train] Timeout Error during backprop: {backprop_error}", "red")
    elif backprop_results and "error" in backprop_results:
        backprop_error = backprop_results["error"]
        backprop_error_type = backprop_results.get("error_type", "general")
        if backprop_error_type == "OOM":
            print_colored(f"[Train] OOM Error during backprop: {backprop_error}", "red")
        else:
            print_colored(f"[Train] Error during backprop: {backprop_error}", "red")
    elif backprop_results and backprop_results["any_updates"]:
        print(f"[Train] Backprop successful! Model saved to {MODEL_PATH}")
        print(f"[Train] Timings: {backprop_results['timings']}")
        CURRENT_LATEST_MODEL_PATH = MODEL_PATH
    else:
        print(f"[Train] No backprop updates applied")
    

    log_entry = {"iteration": iteration, "mean_train_score": mean_train_score, "mean_eval_score": mean_eval_score, "unique_answers": len(unique_answers), "num_eval_responses": len(evaluation_responses), "num_train_responses": len(training_responses), "uniqueness": uniqueness, "correct_logprobs": correct_logprobs, "incorrect_logprobs": incorrect_logprobs, "num_unique_correct_answers": len(unique_correct_answers), "backprop_error": backprop_error, "backprop_error_type": backprop_error_type}
    with open(logs_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    with open(unique_answer_path, "a") as f:
        f.write(json.dumps(unique_correct_answers) + "\n")

    iteration += 1

    if mean_eval_score >= 0.99 or iteration >= args.max_iterations:
        print(f"\n[Train] Completed iteration {iteration}\n")
        break
    print(f"\n[Train] Completed iteration {iteration}\n")
print(f"\n[Train] Completed all iterations\n")
