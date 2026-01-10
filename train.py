from utils_rollout import generate_responses, generate_tree_responses, generate_priority_tree_responses
import argparse, json, torch, time, numpy as np, re, os, tqdm
from utils import print_colored, DoublePrint, get_git_version
from llms.genserv.client import GenerationServiceClient
from evalserv_client import EvaluationServiceClient
from utils_tmux import start_gen_and_eval_sessions
from concurrent.futures import ThreadPoolExecutor
from utils_experiments import make_exp_folder
from backprop_worker import BackpropWorker
from utils_code import canonicalize_code, normalize_whitespace, strip_comments, format_normalize
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
parser.add_argument("--base_model", type=str, default="microsoft/phi-4") # Qwen/Qwen3-14b-base
parser.add_argument("--task_id", type=str, default="sharded-livecodebench/2857")

parser.add_argument("--sample_strategy", type=str, default="tree", choices=["iid", "tree", "prio"])
parser.add_argument("--group_size", type=int, default=100)
parser.add_argument("--tree_degree", type=int, default=2)
parser.add_argument("--tree_depth", type=int, default=13)

parser.add_argument("--num_eval_runs", type=int, default=500)
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())

# Backprop
parser.add_argument("--backprop_method", type=str, default="grpo", choices=["grpo", "kto", "rej"])
parser.add_argument("--kto_margin", type=float, default=None)
parser.add_argument("--advantage_estimation", type=str, default="zero_mean", choices=["zero_mean", "zero_mean_noneg"])
parser.add_argument("--learning_rate", type=float, default=5e-3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_iterations", type=int, default=25)

args = parser.parse_args()

start_gen_and_eval_sessions()

exp_folder = make_exp_folder()
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


def run_evaluation_phase(conversation, num_eval_runs):
    responses = generate_responses(assistant_gen_client, eval_client, sample, conversation, num_eval_runs)
    return responses


def run_training_phase(conversation, sample_strategy, group_size, tree_depth, tree_degree):
    if sample_strategy == "iid":
        responses = generate_responses(assistant_gen_client, eval_client, sample, conversation, group_size)
    elif sample_strategy == "tree":
        responses = generate_tree_responses(assistant_gen_client, eval_client, sample, conversation, tree_depth, tree_degree)
    elif sample_strategy == "prio":
        responses = generate_priority_tree_responses(assistant_gen_client, eval_client, sample, conversation, group_size)
    return responses

system_message = task.generate_system_prompt(sample)
input_prompt = task.populate_fully_specific_prompt(sample)

conversation = [{"role": "system", "content": system_message}, {"role": "user", "content": input_prompt}]

CURRENT_LATEST_MODEL_PATH = args.base_model
iteration = 0

while True:
    iteration_start_time = time.time()
    
    # Step 1: Forward
    # Step 1a: Load the model on vllm backend
    load_start_time = time.time()
    load_result = assistant_gen_client.load_model(CURRENT_LATEST_MODEL_PATH, num_gpus=args.num_gpus) # Be careful, this shouldn't be commented by default
    load_time = time.time() - load_start_time
    # print(f"Model load result: {load_result}")

    # Step 1b: Generate responses
    print(">> Starting evaluation and training phases in parallel")
    training_start_time = time.time()
    evaluation_start_time = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        training_future = executor.submit(run_training_phase, conversation, args.sample_strategy, args.group_size, args.tree_depth, args.tree_degree)
        evaluation_future = executor.submit(run_evaluation_phase, conversation, args.num_eval_runs)
        training_responses = training_future.result()
        training_time = time.time() - training_start_time
        evaluation_responses = evaluation_future.result()
        evaluation_time = time.time() - evaluation_start_time

    # responses = evaluation_responses + training_responses

    for response in training_responses + evaluation_responses:
        response["answer_raw"] = extract_answer(response["response_text"])
        response["answer_ws_norm"] = normalize_whitespace(response["answer_raw"])
        response["answer_no_comment"] = strip_comments(response["answer_raw"])
        response["answer_formatted"] = format_normalize(response["answer_raw"])
        response["answer_ast"] = canonicalize_code(response["answer_raw"])
        response["token_nll"] = - response["logprobs"] / len(response["response_tokens"])

    # compute the uniqueness of the answers at all levels
    unique_raw = set([response["answer_raw"] for response in evaluation_responses])
    unique_ws_norm = set([response["answer_ws_norm"] for response in evaluation_responses])
    unique_no_comment = set([response["answer_no_comment"] for response in evaluation_responses])
    unique_formatted = set([response["answer_formatted"] for response in evaluation_responses])
    unique_ast = set([response["answer_ast"] for response in evaluation_responses])

    unique_correct_answers = sorted(set([response["answer_ast"] for response in evaluation_responses if response["score"] == 1]))

    correct_responses = [response for response in evaluation_responses if response["score"] == 1]
    incorrect_responses = [response for response in evaluation_responses if response["score"] != 1]

    correct_logprobs = np.mean([response["logprobs"] for response in correct_responses])
    incorrect_logprobs = np.mean([response["logprobs"] for response in incorrect_responses])

    correct_token_nll = np.mean([response["token_nll"] for response in correct_responses])
    incorrect_token_nll = np.mean([response["token_nll"] for response in incorrect_responses])

    correct_resp_length = np.mean([len(response["response_tokens"]) for response in correct_responses])
    incorrect_resp_length = np.mean([len(response["response_tokens"]) for response in incorrect_responses])

    mean_train_score = np.mean([response["score"] for response in training_responses])
    mean_eval_score = np.mean([response["score"] for response in evaluation_responses])
    
    # Calculate uniqueness percentages for each level
    uniqueness_raw = 100.0 * len(unique_raw) / len(evaluation_responses)
    uniqueness_ws_norm = 100.0 * len(unique_ws_norm) / len(evaluation_responses)
    uniqueness_no_comment = 100.0 * len(unique_no_comment) / len(evaluation_responses)
    uniqueness_formatted = 100.0 * len(unique_formatted) / len(evaluation_responses)
    uniqueness_ast = 100.0 * len(unique_ast) / len(evaluation_responses)
    
    print_colored(f"Mean train score: {mean_train_score}", "green")
    print_colored(f"Mean eval score: {mean_eval_score}", "green")
    print_colored(f"Uniqueness levels:", "green")
    print_colored(f"  Raw: {uniqueness_raw:.2f}% ({len(unique_raw)}/{len(evaluation_responses)})", "green")
    print_colored(f"  WS-Normalized: {uniqueness_ws_norm:.2f}% ({len(unique_ws_norm)}/{len(evaluation_responses)})", "green")
    print_colored(f"  No-Comments: {uniqueness_no_comment:.2f}% ({len(unique_no_comment)}/{len(evaluation_responses)})", "green")
    print_colored(f"  Formatted: {uniqueness_formatted:.2f}% ({len(unique_formatted)}/{len(evaluation_responses)})", "green")
    print_colored(f"  AST: {uniqueness_ast:.2f}% ({len(unique_ast)}/{len(evaluation_responses)})", "green")

    # Step 1c: Unload the model
    unload_start_time = time.time()
    unload_result = assistant_gen_client.unload_model()
    unload_time = time.time() - unload_start_time
    # print(f"Model unload result: {unload_result}")

    # Step 2: Backprop
    MODEL_PATH = f"{model_save_path}"
    
    backprop_args = {"backprop_method": args.backprop_method, "learning_rate": args.learning_rate, "advantage_estimation": args.advantage_estimation, "batch_size": args.batch_size, "reduction": "sum", "kto_margin": args.kto_margin}
    
    print(f"\n[Train] Starting backprop with {len(training_responses)} responses")
    backprop_start_time = time.time()
    backprop_results = backprop_worker.run_backprop(model_path=CURRENT_LATEST_MODEL_PATH, save_path=MODEL_PATH, conversation=conversation, responses=training_responses, args_dict=backprop_args, timeout=1800)
    backprop_time = time.time() - backprop_start_time

    backprop_results_stats = backprop_results.get("stats", {})
    
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
    

    total_iteration_time = time.time() - iteration_start_time
    
    log_entry = {"iteration": iteration, "mean_train_score": mean_train_score, "mean_eval_score": mean_eval_score, "num_eval_responses": len(evaluation_responses), "num_train_responses": len(training_responses),
    
    "unique_count_raw": len(unique_raw), "unique_count_ws_norm": len(unique_ws_norm), "unique_count_no_comment": len(unique_no_comment), "unique_count_formatted": len(unique_formatted), "unique_count_ast": len(unique_ast),
    "uniqueness_raw": uniqueness_raw, "uniqueness_ws_norm": uniqueness_ws_norm, "uniqueness_no_comment": uniqueness_no_comment, "uniqueness_formatted": uniqueness_formatted, "uniqueness_ast": uniqueness_ast,
    "num_unique_correct_answers": len(unique_correct_answers),
 
    "correct_logprobs": correct_logprobs, "incorrect_logprobs": incorrect_logprobs,
    "correct_resp_length": correct_resp_length, "incorrect_resp_length": incorrect_resp_length,
    "correct_token_nll": correct_token_nll, "incorrect_token_nll": incorrect_token_nll,

    "backprop_error": backprop_error, "backprop_error_type": backprop_error_type, "timings/load_genserv": load_time, "timings/training_phase": training_time, "timings/evaluation_phase": evaluation_time, "timings/unload_genserv": unload_time, "timings/backprop": backprop_time, "timings/total_iteration": total_iteration_time}
    log_entry.update(backprop_results_stats)

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
