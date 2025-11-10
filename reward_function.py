import torch
import time
from typing import List, Dict, Any
from evalserv_client import EvaluationServiceClient
from tasks import get_task

class RewardFunction:
    def __init__(self, eval_service_url: str = "http://localhost:5001", timeout: float = 120.0):
        self.eval_client = EvaluationServiceClient(base_url=eval_service_url)
        self.timeout = timeout
        
    def compute_rewards(self, samples: List[Dict], responses: List[str], task_name: str = "code") -> List[float]:
        rewards = []
        task = get_task(task_name)
        
        for sample, response_text in zip(samples, responses):
            try:
                system_message = task.generate_system_prompt(sample)
                input_prompt = task.populate_fully_specific_prompt(sample)
                conversation = [{"role": "system", "content": system_message}, {"role": "user", "content": input_prompt}, {"role": "assistant", "content": response_text}]
                
                eval_schedule_result = self.eval_client.schedule_evaluation(conversation, sample["task"], sample)
                eval_status = {"status": "pending"}
                start_time = time.time()
                
                while eval_status["status"] != "completed":
                    if time.time() - start_time > self.timeout:
                        print(f"Evaluation timeout for sample {sample.get('task_id', 'unknown')}")
                        rewards.append(0.0)
                        break
                    eval_status = self.eval_client.check_job(eval_schedule_result["job_id"])
                    time.sleep(0.2)
                
                if eval_status["status"] == "completed":
                    if "evaluation_return" in eval_status["result"]:
                        eval_result = eval_status["result"]["evaluation_return"]
                        reward = float(eval_result.get("score", 0.0))
                        rewards.append(reward)
                    else:
                        rewards.append(0.0)
                else:
                    if eval_status["status"] != "pending":
                        rewards.append(0.0)
                        
            except Exception as e:
                print(f"Error evaluating sample: {e}")
                rewards.append(0.0)
        
        return rewards
    
    def batch_compute_rewards(self, samples: List[Dict], responses: List[str], task_name: str = "code", batch_size: int = 32) -> torch.Tensor:
        all_rewards = []
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i+batch_size]
            batch_responses = responses[i:i+batch_size]
            batch_rewards = self.compute_rewards(batch_samples, batch_responses, task_name)
            all_rewards.extend(batch_rewards)
        
        return torch.tensor(all_rewards, dtype=torch.float32)

