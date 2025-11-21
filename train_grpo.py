import argparse, json, random, torch, time, numpy as np, re, os
from llms.genserv.client import GenerationServiceClient
from evalserv_client import EvaluationServiceClient
from utils_tmux import start_gen_and_eval_sessions
from utils_experiments import make_exp_folder
from utils import print_colored, DoublePrint
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
parser.add_argument("--task_id", type=str, default="sharded-HumanEval/76")
parser.add_argument("--group_size", type=int, default=100)
parser.add_argument("--num_eval_runs", type=int, default=1000)
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())

# Backprop
parser.add_argument("--advantage_estimation", type=str, default="zero_mean", choices=["zero_mean", "zero_mean_noneg"])
parser.add_argument("--learning_rate", type=float, default=5e-3)
parser.add_argument("--effective_batch_size", type=int, default=16)

args = parser.parse_args()

start_gen_and_eval_sessions()

exp_folder = make_exp_folder()
print(f"Experiment folder: {exp_folder}")
model_save_path = os.path.join(exp_folder, "model")
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path, exist_ok=True)

args_path = os.path.join(exp_folder, "args.json")
run_params = vars(args)
run_params["experiment_name"] = exp_folder.split("/")[-1]
run_params["training_method"] = "reinforce"
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
    responses = generate_responses(conversation, args.group_size + args.num_eval_runs)

    for response in responses:
        response["answer"] = extract_answer(response["response_text"])
        response["answer2"] = re.sub(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|#.*?$)', '', response["answer"], flags=re.DOTALL | re.MULTILINE)
        response["answer2"] = "\n".join([line for line in response["answer2"].split("\n") if line.strip()]) # remove any empty lines

    random.shuffle(responses)

    train_responses = responses[:args.group_size]
    eval_responses = responses[args.group_size:]

    # compute the uniqueness of the answers
    unique_answers = set([response["answer2"] for response in eval_responses])

    unique_correct_answers = sorted(set([response["answer2"] for response in eval_responses if response["score"] == 1]))

    response_logprobs = [response["logprobs"] for response in responses]
    correct_logprobs = [response["logprobs"] for response in eval_responses if response["score"] == 1]
    incorrect_logprobs = [response["logprobs"] for response in eval_responses if response["score"] != 1]
    # print("RESPONSE LOGPROBS:")
    # print(response_logprobs)

    mean_train_score = np.mean([response["score"] for response in train_responses])
    mean_eval_score = np.mean([response["score"] for response in eval_responses])
    uniqueness = 100.0 * len(unique_answers) / len(eval_responses)
    print_colored(f"Mean train score: {mean_train_score}", "green")
    print_colored(f"Mean eval score: {mean_eval_score} (Uniqueness: {len(unique_answers) / len(eval_responses)} ({uniqueness:.2f}))", "green")

    # Step 1c: Unload the model
    unload_result = assistant_gen_client.unload_model()
    # print(f"Model unload result: {unload_result}")

    # Step 2: Backprop
    MODEL_PATH = f"{model_save_path}"
    
    backprop_args = {"learning_rate": args.learning_rate, "advantage_estimation": args.advantage_estimation, "reduction": "sum", "effective_batch_size": args.effective_batch_size}
    
    print(f"\n[Train] Starting backprop with {len(responses)} responses")
    backprop_results = backprop_worker.run_backprop(model_path=CURRENT_LATEST_MODEL_PATH, save_path=MODEL_PATH, conversation=conversation, responses=train_responses, args_dict=backprop_args, timeout=600)
    
    if backprop_results and backprop_results["any_updates"]:
        print(f"[Train] Backprop successful! Model saved to {MODEL_PATH}")
        print(f"[Train] Timings: {backprop_results['timings']}")
        CURRENT_LATEST_MODEL_PATH = MODEL_PATH
    else:
        print(f"[Train] No backprop updates applied")
    

    log_entry = {"iteration": iteration, "mean_train_score": mean_train_score, "mean_eval_score": mean_eval_score, "unique_answers": len(unique_answers), "num_eval_responses": len(eval_responses), "num_train_responses": len(train_responses), "uniqueness": uniqueness, "correct_logprobs": correct_logprobs, "incorrect_logprobs": incorrect_logprobs, "num_unique_correct_answers": len(unique_correct_answers)}
    with open(logs_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    with open(unique_answer_path, "a") as f:
        f.write(json.dumps(unique_correct_answers) + "\n")

    iteration += 1

    if mean_eval_score >= 0.99 or iteration >= 100:
        print(f"\n[Train] Completed iteration {iteration}\n")
        break
    print(f"\n[Train] Completed iteration {iteration}\n")
print(f"\n[Train] Completed all iterations\n")
