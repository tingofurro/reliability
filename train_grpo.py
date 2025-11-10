import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from datasets import Dataset
from reward_function import RewardFunction
from tasks import get_task
from rl_config import GRPOConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(dataset_path, task_name="code"):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    data = [d for d in data if d["task"] == task_name]
    return data

def prepare_prompt(sample, task_name="code"):
    task = get_task(task_name)
    system_message = task.generate_system_prompt(sample)
    input_prompt = task.populate_fully_specific_prompt(sample)
    full_prompt = f"{system_message}\n\n{input_prompt}"
    return full_prompt

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Phi-4 on Code Tasks")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-4", help="Model to train")
    parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phi4_grpo", help="Output directory")
    parser.add_argument("--eval_service_url", type=str, default="http://localhost:5001", help="Evaluation service URL")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--group_size", type=int, default=8, help="Number of samples per group")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--save_freq", type=int, default=500, help="Save checkpoint frequency")
    parser.add_argument("--log_with", type=str, default="wandb", help="Logging service (wandb, tensorboard)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"=" * 80)
    print(f"GRPO Training Configuration")
    print(f"=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_fn}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Group Size: {args.group_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Steps: {args.max_steps}")
    print(f"=" * 80)
    
    config = GRPOConfig()
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.group_size = args.group_size
    config.max_steps = args.max_steps
    config.save_freq = args.save_freq
    config.output_dir = args.output_dir
    config.log_with = args.log_with
    config.seed = args.seed
    config.warmup_steps = args.warmup_steps
    
    print(f"Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading dataset from {args.dataset_fn}...")
    data = load_data(args.dataset_fn)
    print(f"Loaded {len(data)} samples")
    
    print(f"Preparing dataset for training...")
    dataset_dict = {"sample": [], "prompt": [], "input_ids": [], "task_id": []}
    
    for sample in data:
        prompt = prepare_prompt(sample)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
        
        if len(input_ids) > 1024:
            continue
            
        dataset_dict["sample"].append(sample)
        dataset_dict["prompt"].append(prompt)
        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["task_id"].append(sample.get("task_id", "unknown"))
    
    dataset = Dataset.from_dict(dataset_dict)
    print(f"Prepared {len(dataset)} samples for training")
    
    reward_fn = RewardFunction(eval_service_url=args.eval_service_url)
    
    print(f"Loading model (no value head needed for GRPO)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.train()
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.max_steps)
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
    
    if config.log_with == "wandb":
        try:
            import wandb
            wandb.init(project=config.tracker_project_name, config=vars(args))
        except:
            print("Warning: Could not initialize wandb")
    
    print(f"Starting GRPO training...")
    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    step = 0
    for epoch in range(config.num_train_epochs):
        for batch_idx, batch in enumerate(dataloader):
            if step >= config.max_steps:
                break
            
            prompts = batch["prompt"]
            samples = batch["sample"]
            
            all_query_tensors = []
            all_response_tensors = []
            all_responses = []
            all_samples = []
            
            for prompt, sample in zip(prompts, samples):
                query_tensor = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                
                for _ in range(config.group_size):
                    with torch.no_grad():
                        output = model.generate(query_tensor, **generation_kwargs, return_dict_in_generate=True, output_scores=False)
                    
                    response_tensor = output.sequences[0][query_tensor.shape[1]:]
                    response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
                    
                    all_query_tensors.append(query_tensor)
                    all_response_tensors.append(response_tensor)
                    all_responses.append(response_text)
                    all_samples.append(sample)
            
            rewards = reward_fn.compute_rewards(all_samples, all_responses, task_name="code")
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(model.device)
            
            rewards_grouped = rewards_tensor.view(-1, config.group_size)
            
            if config.whiten_rewards:
                mean_reward = rewards_grouped.mean(dim=1, keepdim=True)
                std_reward = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
                advantages = (rewards_grouped - mean_reward) / std_reward
            else:
                advantages = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)
            
            advantages = advantages.view(-1)
            
            policy_loss = 0
            kl_div = 0
            
            for i, (query_tensor, response_tensor, advantage) in enumerate(zip(all_query_tensors, all_response_tensors, advantages)):
                full_tensor = torch.cat([query_tensor.squeeze(0), response_tensor]).unsqueeze(0)
                
                if full_tensor.shape[1] > 2048:
                    continue
                
                with torch.no_grad():
                    ref_outputs = ref_model(full_tensor)
                    ref_logits = ref_outputs.logits[:, query_tensor.shape[1]-1:-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs.gather(2, response_tensor.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                
                model_outputs = model(full_tensor)
                model_logits = model_outputs.logits[:, query_tensor.shape[1]-1:-1, :]
                model_log_probs = F.log_softmax(model_logits, dim=-1)
                model_token_log_probs = model_log_probs.gather(2, response_tensor.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                
                log_ratio = model_token_log_probs - ref_token_log_probs
                kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()
                kl_div += kl.item()
                
                loss = -advantage * model_token_log_probs.sum() + config.beta * kl
                policy_loss += loss
            
            policy_loss = policy_loss / len(all_query_tensors)
            
            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            if step % config.logging_steps == 0:
                avg_reward = rewards_tensor.mean().item()
                avg_kl = kl_div / len(all_query_tensors)
                print(f"Step {step}: Loss={policy_loss.item():.4f}, Reward={avg_reward:.4f}, KL={avg_kl:.4f}")
                
                if config.log_with == "wandb":
                    try:
                        import wandb
                        wandb.log({"loss": policy_loss.item(), "reward": avg_reward, "kl_div": avg_kl, "step": step, "learning_rate": scheduler.get_last_lr()[0]})
                    except:
                        pass
            
            if step % config.save_freq == 0 and step > 0:
                save_dir = Path(config.output_dir) / f"checkpoint-{step}"
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"Saved checkpoint to {save_dir}")
            
            step += 1
            
            if step >= config.max_steps:
                break
    
    print(f"Training complete! Saving final model...")
    final_dir = Path(config.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training finished!")

if __name__ == "__main__":
    main()

