import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from reward_function import RewardFunction
from tasks import get_task
from rl_config import PPOConfig as CustomPPOConfig

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
    parser = argparse.ArgumentParser(description="PPO Training for Phi-4 on Code Tasks")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-4", help="Model to train")
    parser.add_argument("--dataset_fn", type=str, default="data/sharded_instructions_600.json", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phi4_ppo", help="Output directory")
    parser.add_argument("--eval_service_url", type=str, default="http://localhost:5001", help="Evaluation service URL")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--mini_batch_size", type=int, default=4, help="Mini batch size for PPO updates")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--save_freq", type=int, default=500, help="Save checkpoint frequency")
    parser.add_argument("--log_with", type=str, default="wandb", help="Logging service (wandb, tensorboard)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--sample_id", type=str, default="sharded-livecodebench/2877", help="Task ID of the sample to train on")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"=" * 80)
    print(f"PPO Training Configuration")
    print(f"=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_fn}")
    print(f"Sample ID: {args.sample_id}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Steps: {args.max_steps}")
    print(f"=" * 80)
    
    config = CustomPPOConfig()
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.mini_batch_size = args.mini_batch_size
    config.max_steps = args.max_steps
    config.save_freq = args.save_freq
    config.output_dir = args.output_dir
    config.log_with = args.log_with
    config.seed = args.seed
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    print(f"Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading dataset from {args.dataset_fn}...")
    data = load_data(args.dataset_fn)
    print(f"Loaded {len(data)} samples")
    
    # Filter for the specific sample_id
    data = [d for d in data if d.get("task_id") == args.sample_id]
    if len(data) == 0:
        raise ValueError(f"No sample found with task_id: {args.sample_id}")
    print(f"Training on single sample with task_id: {args.sample_id}")
    
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
    
    print(f"Initializing PPO training...")
    
    trl_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        max_grad_norm=config.max_grad_norm,
        seed=config.seed,
        log_with=config.log_with,
        tracker_project_name=config.tracker_project_name,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=config.target_kl,
        init_kl_coef=config.init_kl_coef,
        adap_kl_ctrl=config.adap_kl_ctrl,
        cliprange=config.cliprange,
        cliprange_value=config.cliprange_value,
        vf_coef=config.vf_coef,
        gamma=config.gamma,
        lam=config.lam,
    )
    
    print(f"Loading model with value head...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    trainer = PPOTrainer(
        config=trl_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )
    
    print(f"Starting PPO training...")
    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "do_sample": config.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    for epoch, batch in enumerate(trainer.dataloader):
        query_tensors = batch["input_ids"]
        
        response_tensors = trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs
        )
        
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        rewards = reward_fn.batch_compute_rewards(
            batch["sample"],
            batch["response"],
            task_name="code"
        )
        
        stats = trainer.step(query_tensors, response_tensors, rewards)
        trainer.log_stats(stats, batch, rewards)
        print(f"Epoch {epoch}: Mean Reward = {rewards.mean():.4f}, Min = {rewards.min():.4f}, Max = {rewards.max():.4f}")
        
        if epoch % config.save_freq == 0 and epoch > 0:
            save_dir = Path(config.output_dir) / f"checkpoint-{epoch}"
            save_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_pretrained(save_dir)
            print(f"Saved checkpoint to {save_dir}")
        
        if epoch >= config.max_steps:
            break
    
    print(f"Training complete! Saving final model...")
    final_dir = Path(config.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(final_dir)
    print(f"Training finished!")

if __name__ == "__main__":
    main()

