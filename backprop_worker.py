import multiprocessing, torch, numpy as np, setproctitle, os, time, traceback, sys
from model_generator_hf import GenerationModel
from utils import print_colored


class TeeOutput:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)
        self.file1.flush()
        self.file2.flush()

    def flush(self):
        self.file1.flush()
        self.file2.flush()


def backprop_worker_process(model_path, save_path, conversation, responses, args_dict, result_queue, error_queue):
    setproctitle.setproctitle("backprop_worker")
    
    print(f"[Backprop Worker] Starting backprop worker (PID: {os.getpid()})")
    
    timings = {"model_load": 0, "backprop": 0, "model_save": 0}
    
    reduction = args_dict.get("reduction", "sum")
    advantage_estimation = args_dict.get("advantage_estimation", "zero_mean")
    effective_batch_size = args_dict.get("effective_batch_size", 16)

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
        
        advantages = torch.tensor(advantages).to(assistant_model.device)
        print(f"[Backprop Worker] Advantages computed using {advantage_estimation}")
        print(f"[Backprop Worker] Advantages: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}")
        
        # Filter out responses with zero advantage if using zero_mean_noneg
        if advantage_estimation == "zero_mean_noneg":
            selected_indices = [i for i, adv in enumerate(advantages) if adv > 0]
            if len(selected_indices) == 0:
                print_colored("[Backprop Worker] No responses with positive advantage, skipping backprop", "yellow")
                result_queue.put({"any_updates": False, "losses": [], "timings": timings, "num_responses": len(responses)})
                return
            selected_responses = [responses[i] for i in selected_indices]
            selected_advantages = advantages[selected_indices]
        else:
            selected_responses = responses
            selected_advantages = advantages
        
        print(f"[Backprop Worker] Using {len(selected_responses)} responses for backprop")
        print(f"[Backprop Worker] Using effective batch size of {effective_batch_size}")
        
        # Process responses in batches with gradient accumulation
        num_batches = (len(selected_responses) + effective_batch_size - 1) // effective_batch_size
        print(f"[Backprop Worker] Processing {num_batches} batches")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * effective_batch_size
            end_idx = min(start_idx + effective_batch_size, len(selected_responses))
            batch_responses = selected_responses[start_idx:end_idx]
            batch_advantages = selected_advantages[start_idx:end_idx]
            
            print(f"[Backprop Worker] Processing batch {batch_idx + 1}/{num_batches} (responses {start_idx}-{end_idx})")
            
            # print(conversation)
            # Get logprobs for this batch
            batch_logprobs = []
            for response in batch_responses:
                logprob = assistant_model.get_logprobs(conversation, [response], reduction=reduction)[0]
                # print(logprob, response["logprobs"])
                batch_logprobs.append(logprob)
            
            batch_logprobs = torch.stack(batch_logprobs)
            
            # Check for unstable logprobs
            if any(logprob < -1000 for logprob in batch_logprobs):
                print_colored(f"[Backprop Worker] Batch {batch_idx + 1} has unstable logprobs, skipping", "yellow")
                continue
            
            # Compute loss for this batch (normalized by batch size)
            batch_loss = -torch.sum(batch_advantages * batch_logprobs) / num_batches
            
            # Backward pass - accumulates gradients
            optimizer.zero_grad()
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(assistant_model.model.parameters(), max_norm=4.0)
            optimizer.step()
            any_updates = True
            print_colored("[Backprop Worker] Backprop update applied successfully", "green")

            # Clear tensors to save memory
            del batch_logprobs, batch_loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"[Backprop Worker] Batch {batch_idx + 1}/{num_batches} completed")
        
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
    except Exception as e:
        print(f"[Backprop Worker] Error in backprop: {e}")
        error_queue.put({"error": str(e), "traceback": traceback.format_exc()})
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
            return None
        
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

