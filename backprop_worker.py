from llms.genserv.utils_prefix_tree import build_prefix_tree, calculate_backtrack_scores, get_backprop_ops, visualize_prefix_tree
import multiprocessing, torch, numpy as np, setproctitle, os, time, traceback, sys
from model_generator_hf import GenerationModel
from utils import TeeOutput, print_colored

def calculate_gradients_grpo(assistant_model, conversation, responses, args_dict):
    reduction = args_dict.get("reduction", "sum")
    advantage_estimation = args_dict.get("advantage_estimation", "zero_mean")
    batch_size = args_dict.get("batch_size", 16)

    # Extract scores and compute advantages
    scores = np.array([response["score"] for response in responses])
    print(f"[Backprop Worker] Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

    if advantage_estimation == "zero_mean":
        advantages = scores - scores.mean()
    elif advantage_estimation == "zero_mean_noneg":
        advantages = scores - scores.mean()
        advantages = np.maximum(0, advantages)
    else:
        raise ValueError(f"Unknown advantage_estimation: {advantage_estimation}")

    print(f"[Backprop Worker] Advantages computed using {advantage_estimation}")
    print(f"[Backprop Worker] Advantages: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}")        

    for response, advantage in zip(responses, advantages):
        response["advantage"] = advantage

    # Filter out responses with zero advantage if using zero_mean_noneg
    selected_responses = [response for response in responses if response["logprobs"] > -1000]
    if advantage_estimation == "zero_mean_noneg":
        selected_responses = [response for response in selected_responses if response["advantage"] > 0]

    selected_advantages = torch.tensor([response["advantage"] for response in selected_responses]).to(assistant_model.device)

    if len(selected_responses) == 0:
        print_colored("[Backprop Worker] No responses with positive advantage, skipping backprop", "yellow")
        return None

    num_responses = len(selected_responses)
    print(f"[Backprop Worker] Using {num_responses} responses for backprop")
    
    # Calculate number of gradient accumulation steps based on fixed batch size
    num_steps = (num_responses + batch_size - 1) // batch_size
    print(f"[Backprop Worker] Using batch_size={batch_size}, gradient accumulation steps={num_steps}")
    print(f"[Backprop Worker] Processing {num_steps} batches of up to {batch_size} responses each")

    # Process responses in batches with gradient accumulation
    for step_idx in range(num_steps):
        start_idx = step_idx * batch_size
        end_idx = min(start_idx + batch_size, num_responses)
        batch_responses = selected_responses[start_idx:end_idx]
        batch_advantages = selected_advantages[start_idx:end_idx]
        
        print(f"[Backprop Worker] Processing batch {step_idx + 1}/{num_steps} (responses {start_idx}-{end_idx})")
        
        # Get logprobs for this batch
        batch_logprobs = []
        for response in batch_responses:
            logprob = assistant_model.get_logprobs(conversation, [response], reduction=reduction)[0]
            batch_logprobs.append(logprob)
        
        batch_logprobs = torch.stack(batch_logprobs)
        
        # Compute loss for this batch (normalized by total number of responses)
        batch_loss = -torch.sum(batch_advantages * batch_logprobs) / num_responses
        
        # Backward pass - accumulates gradients
        batch_loss.backward()
        
        # Clear tensors to save memory
        del batch_logprobs, batch_loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"[Backprop Worker] Batch {step_idx + 1}/{num_steps} completed")
    return {"success": True}


def calculate_gradients_kto(assistant_model, conversation, responses, args_dict):
    prefix_tree = build_prefix_tree(responses)
    calculate_backtrack_scores(prefix_tree)
    backprop_ops = get_backprop_ops(prefix_tree)

    print(f"[Backprop Worker] Found {len(backprop_ops)} backprop operations")

    for op_idx, backprop_op in enumerate(backprop_ops):
        prefix = backprop_op["prefix"]
        options = backprop_op["options"]
        branch_token_ids = [option["branch_tokens"][0] for option in options]

        branch_logprobs = assistant_model.get_branch_logprobs(conversation, prefix, branch_token_ids)
        advantages = [option["advantage"] for option in options]
        advantages = torch.tensor(advantages).to(assistant_model.device)

        chunk_loss = -torch.sum(advantages * branch_logprobs)
        chunk_loss.backward()

        del branch_logprobs, chunk_loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"[Backprop Worker] Backprop operation {op_idx + 1}/{len(backprop_ops)} completed")

    return {"success": True}


def calculate_gradients_sft(assistant_model, conversation, responses, args_dict):
    reduction = args_dict.get("reduction", "sum")
    batch_size = args_dict.get("batch_size", 16)
    
    # Filter responses with score of 1
    correct_responses = [response for response in responses if response.get("score") == 1]
    print(f"[Backprop Worker] Found {len(correct_responses)} responses with score=1 out of {len(responses)} total")
    
    if len(correct_responses) == 0:
        print_colored("[Backprop Worker] No responses with score=1, skipping SFT", "yellow")
        return None
    
    # Deduplicate responses by text content
    seen_texts = set()
    unique_responses = []
    for response in correct_responses:
        response_text = response.get("response_text", "")
        if response_text and response_text not in seen_texts:
            seen_texts.add(response_text)
            unique_responses.append(response)
    
    num_responses = len(unique_responses)
    print(f"[Backprop Worker] Using {num_responses} unique correct responses for SFT")
    
    if num_responses == 0:
        print_colored("[Backprop Worker] No unique correct responses, skipping SFT", "yellow")
        return None
    
    # Calculate number of gradient accumulation steps based on fixed batch size
    num_steps = (num_responses + batch_size - 1) // batch_size
    print(f"[Backprop Worker] Using batch_size={batch_size}, gradient accumulation steps={num_steps}")
    print(f"[Backprop Worker] Processing {num_steps} batches of up to {batch_size} responses each")
    
    # Process responses in batches with gradient accumulation
    for step_idx in range(num_steps):
        start_idx = step_idx * batch_size
        end_idx = min(start_idx + batch_size, num_responses)
        batch_responses = unique_responses[start_idx:end_idx]
        
        print(f"[Backprop Worker] Processing batch {step_idx + 1}/{num_steps} (responses {start_idx}-{end_idx})")
        
        # Get logprobs for this batch
        batch_logprobs = []
        for response in batch_responses:
            logprob = assistant_model.get_logprobs(conversation, [response], reduction=reduction)[0]
            batch_logprobs.append(logprob)
        
        batch_logprobs = torch.stack(batch_logprobs)
        
        # Compute loss for this batch (normalized by total number of responses)
        batch_loss = -torch.sum(batch_logprobs) / num_responses
        
        # Backward pass - accumulates gradients
        batch_loss.backward()
        
        # Clear tensors to save memory
        del batch_logprobs, batch_loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"[Backprop Worker] Batch {step_idx + 1}/{num_steps} completed")
    
    return {"success": True}

def backprop_worker_process(model_path, save_path, conversation, responses, args_dict, result_queue, error_queue):
    setproctitle.setproctitle("backprop_worker")
    
    print(f"[Backprop Worker] Starting backprop worker (PID: {os.getpid()})")
    
    timings = {"model_load": 0, "backprop": 0, "model_save": 0}
    
    backprop_method = args_dict.get("backprop_method", "grpo")
    reduction = args_dict.get("reduction", "sum")
    advantage_estimation = args_dict.get("advantage_estimation", "zero_mean")
    batch_size = args_dict.get("batch_size", 16)

    # Load model and optimizer
    T_model_load_start = time.time()
    print(f"[Backprop Worker] Loading model from {model_path}")
    assistant_model = GenerationModel(model_name=model_path, device=None)
    optimizer = torch.optim.SGD(assistant_model.model.parameters(), lr=args_dict["learning_rate"])
    T_model_load_end = time.time()
    timings["model_load"] = T_model_load_end - T_model_load_start
    
    print(f"[Backprop Worker] Model loaded successfully in {timings['model_load']:.2f}s")
    
    log_file = open('backprop_worker.log', 'a')
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)
    
    any_updates = False
    losses = []
    
    print(f"[Backprop Worker] Processing {len(responses)} responses for backprop")
    
    try:
        T_backprop_start = time.time()
        
        # Zero gradients once at the start
        optimizer.zero_grad()

        if backprop_method == "grpo":
            grad_return = calculate_gradients_grpo(assistant_model, conversation, responses, args_dict)
        elif backprop_method == "kto":
            grad_return = calculate_gradients_kto(assistant_model, conversation, responses, args_dict)
        elif backprop_method == "sft":
            grad_return = calculate_gradients_sft(assistant_model, conversation, responses, args_dict)
        else:
            raise ValueError(f"Unknown backprop method: {backprop_method}")
        
        if grad_return is None:
            result_queue.put({"any_updates": False, "losses": [], "timings": timings, "num_responses": len(responses)})
            return None

        # Single optimizer step after all gradient accumulation
        torch.nn.utils.clip_grad_norm_(assistant_model.model.parameters(), max_norm=4.0)
        optimizer.step()
        any_updates = True
        print_colored("[Backprop Worker] Backprop update applied successfully", "green")
        
        T_backprop_end = time.time()
        timings["backprop"] = T_backprop_end - T_backprop_start
        
        # Save model if any updates were made
        if any_updates:
            print(f"[Backprop Worker] Saving updated model to {save_path}")
            assistant_model.save_model(save_path)
            print(f"[Backprop Worker] Model saved successfully")
            T_model_save_end = time.time()
            timings["model_save"] = T_model_save_end - T_backprop_end
        
        # Prepare results
        results = {"any_updates": any_updates, "losses": losses, "timings": timings, "num_responses": len(responses)}
        
        # Send results back
        result_queue.put(results)
    except torch.OutOfMemoryError as e:
        print_colored(f"[Backprop Worker] OOM Error in backprop: {e}", "red")
        error_queue.put({"error": str(e), "error_type": "OOM", "traceback": traceback.format_exc()})
    except Exception as e:
        print(f"[Backprop Worker] Error in backprop: {e}")
        error_queue.put({"error": str(e), "error_type": "general", "traceback": traceback.format_exc()})
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    print(f"[Backprop Worker] Backprop completed successfully")


class BackpropWorker:
    def __init__(self):
        self.process = None
        self.result_queue = None
        self.error_queue = None
    
    def run_backprop(self, model_path, save_path, conversation, responses, args_dict, timeout=300):
        self.result_queue = multiprocessing.Queue()
        self.error_queue = multiprocessing.Queue()
        
        self.process = multiprocessing.Process(target=backprop_worker_process, args=(model_path, save_path, conversation, responses, args_dict, self.result_queue, self.error_queue), daemon=False)
        
        print(f"[Backprop Manager] Starting backprop worker process")
        self.process.start()
        
        self.process.join(timeout=timeout)
        
        if self.process.is_alive():
            print(f"[Backprop Manager] Backprop worker timed out, terminating")
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                print(f"[Backprop Manager] Force killing backprop worker")
                self.process.kill()
                self.process.join()
            return None
        
        if not self.error_queue.empty():
            error_info = self.error_queue.get()
            print(f"[Backprop Manager] Error in backprop worker: {error_info['error']}")
            print(f"[Backprop Manager] Traceback: {error_info['traceback']}")
            return {"any_updates": False, "error": error_info["error"], "error_type": error_info.get("error_type", "general"), "traceback": error_info["traceback"]}
        
        if not self.result_queue.empty():
            results = self.result_queue.get()
            print(f"[Backprop Manager] Backprop completed successfully")
            return results
        else:
            print(f"[Backprop Manager] No results received from backprop worker")
            return None
    
    def cleanup(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()

        if self.result_queue:
            self.result_queue.close()
        if self.error_queue:
            self.error_queue.close()
