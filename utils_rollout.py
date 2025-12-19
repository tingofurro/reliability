import time

def generate_responses(assistant_gen_client, eval_client, sample, conversation, group_size):
    # Schedule all generation jobs in batch
    jobs = [{"conversation": conversation, "n_responses": 1} for _ in range(group_size)]
    batch_result = assistant_gen_client.schedule_job_batch(jobs)
    active_job_ids = batch_result["job_ids"]
    
    responses = []
    eval_job_id2response = {}
    active_eval_job_ids = []

    while active_job_ids or active_eval_job_ids:
        # Check all active generation jobs in batch
        if active_job_ids:
            gen_batch_results = assistant_gen_client.check_job_batch(active_job_ids)
            completed_gen_jobs = []
            evaluations_to_schedule = []
            
            for job_id, job_result in gen_batch_results["results"].items():
                if job_result["status"] == "completed":
                    response = job_result["responses"][0]
                    completed_gen_jobs.append(job_id)
                    this_conversation = conversation + [{"role": "assistant", "content": response["response_text"]}]
                    evaluations_to_schedule.append({"conversation": this_conversation, "task_name": sample["task"], "sample": sample})
                    responses.append(response)
            
            # Schedule evaluation jobs in batch
            if evaluations_to_schedule:
                eval_batch_result = eval_client.schedule_evaluation_batch(evaluations_to_schedule)
                eval_job_ids = eval_batch_result["job_ids"]
                for eval_job_id, response in zip(eval_job_ids, responses[-len(eval_job_ids):]):
                    eval_job_id2response[eval_job_id] = response
                    active_eval_job_ids.append(eval_job_id)
            
            # Remove completed generation jobs
            for job_id in completed_gen_jobs:
                active_job_ids.remove(job_id)
        
        # Check all active evaluation jobs in batch
        if active_eval_job_ids:
            eval_batch_results = eval_client.check_job_batch(active_eval_job_ids)
            completed_eval_jobs = []
            
            for job_id, job_result in eval_batch_results["results"].items():
                if job_result["status"] == "completed" and "evaluation_return" in job_result.get("result", {}):
                    completed_eval_jobs.append(job_id)
                    response = eval_job_id2response[job_id]
                    response["score"] = job_result["result"]["evaluation_return"]["score"]
                elif job_result["status"] == "error" or (job_result["status"] == "completed" and "evaluation_return" not in job_result.get("result", {})):
                    completed_eval_jobs.append(job_id)
                    response = eval_job_id2response[job_id]
                    response["score"] = 0
            
            # Remove completed evaluation jobs
            for job_id in completed_eval_jobs:
                active_eval_job_ids.remove(job_id)
        
        time.sleep(0.1)
    return responses

def generate_tree_responses(assistant_gen_client, eval_client, sample, conversation, depth, degree):
    T = time.time()
    resp = assistant_gen_client.build_tree(conversation, depth=depth, degree=degree)
    job_id = resp["job_id"]

    status = "pending"
    full_tree = []
    active_eval_job_ids = []
    eval_job_id2response = {}
    while status not in  ["completed", "error", "not_found"]:
        time.sleep(5.0)
        resp = assistant_gen_client.check_on_tree(job_id, only_new=True)
        status = resp["status"]
        if status not in ["in_progress", "completed"]:
            # print it in red
            print(f"\033[91m{status} returned in the tree gen! Unexpected\033[0m")
        new_nodes = resp["tree"]
        full_tree += new_nodes
        print(f"Tree building status: {status}; Number of tree nodes: {len(new_nodes)} (total: {len(full_tree)} == {resp['total_nodes_count']}; Time: {time.time() - T:.2f} seconds) ")
        # if len(new_nodes) > 0:

    print(f"\033[92mTree building completed in {time.time() - T:.2f} seconds\033[0m")

    T_eval_start = time.time()
    evaluations = [{"conversation": conversation + [{"role": "assistant", "content": response["response_text"]}], "task_name": sample["task"], "sample": sample} for response in full_tree]
    batch_result = eval_client.schedule_evaluation_batch(evaluations)
    for i, eval_job_id in enumerate(batch_result["job_ids"]):
        active_eval_job_ids.append(eval_job_id)
        eval_job_id2response[eval_job_id] = full_tree[i]

    print(f"Starting to collect evaluation results... (T={time.time() - T:.2f} seconds)")

    active_eval_job_ids = set(active_eval_job_ids)
    while len(active_eval_job_ids) > 0:
        # print(f"Number of active evaluation jobs: {len(active_eval_job_ids)} (T={time.time() - T:.2f} seconds)")
        time.sleep(2.0)
        current_active_eval_job_ids = list(active_eval_job_ids) # make copy as set will change size during iteration
        batch_results = eval_client.check_job_batch(current_active_eval_job_ids)

        for job_id in current_active_eval_job_ids:
            job_result = batch_results["results"][job_id]
            if job_result["status"] == "completed" and "evaluation_return" in job_result["result"]:
                response = eval_job_id2response[job_id]
                response["score"] = job_result["result"]["evaluation_return"]["score"]

                active_eval_job_ids.remove(job_id)
            elif job_result["status"] == "error" or (job_result["status"] == "completed" and "evaluation_return" not in job_result["result"]):
                active_eval_job_ids.remove(job_id)
                response = eval_job_id2response[job_id]
                response["score"] = 0
    print(f"\033[92mAll evaluation results collected in {time.time() - T_eval_start:.2f} seconds\033[0m")
    return full_tree
