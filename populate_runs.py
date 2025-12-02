from llms.genserv.client import GenerationServiceClient
from evalserv_client import EvaluationServiceClient
import argparse, json, torch, random, tqdm, time
from tasks.code.task_code import TaskCode
from concurrent.futures import ThreadPoolExecutor
from utils_logs import get_log_counts, log_single_run
from llms import generate
from tasks import get_task

parser = argparse.ArgumentParser()

# Basics
parser.add_argument("--dataset_fn", type=str, default="data/all_code_samples.json")
# use the + to allow multiple model names
parser.add_argument("--model_name", type=str, default="gs-microsoft/phi-4")
parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
parser.add_argument("--num_runs", type=int, default=100)
parser.add_argument("--workers", type=int, default=40)

logs_path = f"logs/logs.jsonl"

args = parser.parse_args()

assistant_gen_client = GenerationServiceClient(base_url=f"http://localhost:5000")
eval_client = EvaluationServiceClient(base_url=f"http://localhost:5001")

# if any of the models start with gs-, then there should be only one model name
if args.model_name.startswith("gs-"):
    assistant_gen_client.load_model(args.model_name[3:], num_gpus=args.num_gpus, workers_per_gpu=1, max_concurrent_jobs_per_worker=args.workers // args.num_gpus)
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
    task = TaskCode()
    task = get_task(sample["task"])

    system_message = task.generate_system_prompt(sample)
    input_prompt = task.populate_fully_specific_prompt(sample)

    conversation = [{"role": "system", "content": system_message}, {"role": "user", "content": input_prompt}]

    # job_schedule_result = assistant_gen_client.schedule_job(conversation, n_responses=1)
    
    # job_status = {"status": "pending"}
    # while job_status["status"] != "completed":
    #     job_status = assistant_gen_client.check_job(job_schedule_result["job_id"])
    #     time.sleep(0.2)
    
    # response = job_status["responses"][0]
    response_text = generate(conversation, model=args.model_name, max_tokens=2000)

    this_conversation = conversation + [{"role": "assistant", "content": response_text}]

    # this is useful for debugging
    # try:
    #     extracted_answer = task.extract_answer(response_text)
    #     eval_result = task.evaluator_function(extracted_answer, sample)
    # # if it's keyboard interrupt, then don't catch it
    # except KeyboardInterrupt:
    #     raise KeyboardInterrupt
    # except Exception as e:
    #     # print it in red
    #     print(f"\033[91mError with evaluation: {e}\033[0m")
    #     return
    # log_single_run(logs_path, {"task_id": sample["task_id"], "model_name": args.model_name, "response": response_text, "eval_result": eval_result})

    eval_schedule_result = eval_client.schedule_evaluation(this_conversation, sample["task"], sample)
    eval_status = {"status": "pending"}
    while eval_status["status"] != "completed":
        eval_status = eval_client.check_job(eval_schedule_result["job_id"])
        time.sleep(0.2)
    if "evaluation_return" in eval_status["result"]:
        eval_result = eval_status["result"]["evaluation_return"]
        log_single_run(logs_path, {"task_id": sample["task_id"], "model_name": args.model_name, "response": response_text, "eval_result": eval_result})


print(f">> Number of workers: {args.workers}")

with ThreadPoolExecutor(max_workers=args.workers) as executor:
    list(tqdm.tqdm(executor.map(populate_single_run, todo_samples), total=len(todo_samples), desc="Populating runs"))
