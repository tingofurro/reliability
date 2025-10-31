# Fix MKL threading layer compatibility issue
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

# Import numpy first to avoid MKL conflicts
import numpy as np

# from utils_minitree import reconstitute_conversation, merge_responses, find_split_indeces
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import AsyncLLMEngine, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import uuid, json, asyncio, logging, time
from transformers import AutoTokenizer

class AsyncVLLMGenerationModel:
    def __init__(self, model_name="microsoft/phi-4", device=None, 
                 enable_prefix_caching=True, max_context_length=6000, **vllm_kwargs):
        # Configure logging to suppress verbose vLLM messages
        logging.getLogger("vllm.async_llm").setLevel(logging.WARNING)
        logging.getLogger("vllm").setLevel(logging.WARNING)
        
        # Store configuration
        self.model_name = model_name
        self.enable_prefix_caching = enable_prefix_caching
        self.max_context_length = max_context_length
        
        # Truncation configuration options
        self.enable_truncation = os.getenv('VLLM_ENABLE_TRUNCATION', 'true').lower() == 'true'
        self.truncation_warning_threshold = int(os.getenv('VLLM_TRUNCATION_WARNING_THRESHOLD', '5500'))
        self.truncation_verbose = os.getenv('VLLM_TRUNCATION_VERBOSE', 'true').lower() == 'true'

        gpu_memory_utilization = float(os.getenv('GPU_MEMORY_UTILIZATION', '0.96'))

        # Default vLLM engine arguments
        default_vllm_kwargs = {"dtype": "float16", "trust_remote_code": True, "enable_prefix_caching": enable_prefix_caching, "max_model_len": max_context_length, "enforce_eager": False, "disable_custom_all_reduce": True, "enable_chunked_prefill": False, "gpu_memory_utilization": gpu_memory_utilization}
        default_vllm_kwargs.update(vllm_kwargs)
        
        # Check if model exists in shared memory first
        load_path = f"/dev/shm/{model_name}" if os.path.exists(f"/dev/shm/{model_name}") else model_name
        
        # Create engine args and initialize async engine
        engine_args = AsyncEngineArgs(model=load_path, **default_vllm_kwargs)
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Async vLLM model loaded: {model_name} (prefix caching: {'enabled' if enable_prefix_caching else 'disabled'}, max context: {max_context_length})")

    def construct_inputs(self, conversation):
        # First, try to build the prompt normally
        is_last_assistant_message = conversation[-1]['role'] == 'assistant'
        
        # Check if we need to truncate the conversation
        truncated_conversation = self._truncate_conversation_if_needed(conversation)
        
        prompt = self.tokenizer.apply_chat_template(truncated_conversation, tokenize=False)

        # if the last message is an assistant message, then treat that as a prefix
        if not is_last_assistant_message:
            prompt += "<|im_start|>assistant<|im_sep|>"
        else:
            if prompt.endswith(self.tokenizer.eos_token): # need to remove the end of assistant message
                prompt = prompt[:-len(self.tokenizer.eos_token)]
        return prompt

    def _truncate_conversation_if_needed(self, conversation):
        # Build initial prompt to check length
        test_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        tokens = self.tokenizer.encode(test_prompt)
        
        # Add safety margin for chat template additions (assistant prefix, etc.)
        safety_margin = int(os.getenv('VLLM_TRUNCATION_SAFETY_MARGIN', '20'))
        effective_limit = self.max_context_length - safety_margin
        
        # Check if we're approaching the warning threshold
        if len(tokens) > self.truncation_warning_threshold and self.truncation_verbose:
            print(f"Warning: Conversation length ({len(tokens)} tokens) approaching limit ({self.max_context_length})")
        
        # If within effective limit (with safety margin), return original conversation
        if len(tokens) <= effective_limit:
            return conversation
        
        # Check if truncation is enabled
        if not self.enable_truncation:
            if self.truncation_verbose:
                print(f"ERROR: Conversation length ({len(tokens)} tokens) exceeds limit ({self.max_context_length}) but truncation is disabled")
            return conversation  # Return original and let vLLM handle the error
        
        if self.truncation_verbose:
            print(f"Warning: Conversation length ({len(tokens)} tokens) exceeds effective limit ({effective_limit}). Truncating with {safety_margin} token safety margin...")
        
        # Separate messages by role
        system_messages = [msg for msg in conversation if msg['role'] == 'system']
        user_messages = [msg for msg in conversation if msg['role'] == 'user']
        assistant_messages = [(i, msg) for i, msg in enumerate(conversation) if msg['role'] == 'assistant']
        
        # Calculate tokens needed to remove (using effective limit with safety margin)
        excess_tokens = len(tokens) - effective_limit
        tokens_removed = 0
        
        # Create a working copy of the conversation
        truncated_conversation = conversation.copy()
        
        # Sort assistant messages by their original index (oldest first)
        assistant_messages.sort(key=lambda x: x[0])
        
        # Truncate assistant messages starting from the oldest
        for orig_idx, assistant_msg in assistant_messages:
            if tokens_removed >= excess_tokens:
                break
                
            # Find current index in truncated_conversation
            current_idx = None
            for i, msg in enumerate(truncated_conversation):
                if msg is assistant_msg:
                    current_idx = i
                    break
            
            if current_idx is None:
                continue  # Message already removed
            
            # Calculate tokens in this assistant message
            msg_tokens = self.tokenizer.encode(assistant_msg['content'])
            
            if len(msg_tokens) + tokens_removed <= excess_tokens:
                # Remove entire message
                truncated_conversation.pop(current_idx)
                tokens_removed += len(msg_tokens)
                if self.truncation_verbose:
                    print(f"Removed entire assistant message ({len(msg_tokens)} tokens)")
            else:
                # Truncate from the beginning of the message
                tokens_to_remove_from_msg = excess_tokens - tokens_removed
                remaining_tokens = msg_tokens[tokens_to_remove_from_msg:]
                truncated_content = self.tokenizer.decode(remaining_tokens, skip_special_tokens=True)
                
                # Update the message content
                truncated_conversation[current_idx] = {
                    **assistant_msg,
                    'content': truncated_content
                }
                tokens_removed += tokens_to_remove_from_msg
                if self.truncation_verbose:
                    print(f"Truncated assistant message by {tokens_to_remove_from_msg} tokens")
                break
        
        # Verify the truncated conversation is within limits
        final_prompt = self.tokenizer.apply_chat_template(truncated_conversation, tokenize=False)
        final_tokens = self.tokenizer.encode(final_prompt)
        
        if len(final_tokens) > effective_limit:
            if self.truncation_verbose:
                print(f"Warning: Truncation may not be sufficient. Final length: {len(final_tokens)} tokens (effective limit: {effective_limit})")
        else:
            if self.truncation_verbose:
                print(f"Successfully truncated conversation to {len(final_tokens)} tokens (effective limit: {effective_limit}, safety margin: {safety_margin})")
        
        return truncated_conversation

    async def generate_single(self, conversation, temperature=1.0, max_tokens=1000, logprobs=1, schema=None, **kwargs):
        """Generate a single response for a conversation"""
        prompt = self.construct_inputs(conversation)
        request_id = str(uuid.uuid4())
        
        # Set up guided decoding if schema is provided
        if schema:
            guided_decoding_params = GuidedDecodingParams(json=schema)
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=1, logprobs=logprobs, guided_decoding=guided_decoding_params, **kwargs)
        else:
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=1, logprobs=logprobs, **kwargs)
        
        # Generate using async engine
        final_output = None
        async for request_output in self.engine.generate(prompt, sampling_params, request_id):
            final_output = request_output
        
        if final_output and final_output.outputs:
            output = final_output.outputs[0]
            response_text = output.text
            response_tokens = output.token_ids

            cumulative_logprob = output.cumulative_logprob if hasattr(output, 'cumulative_logprob') else None
            per_token_logprobs_obj = output.logprobs if hasattr(output, 'logprobs') else None

            per_token_logprobs = []
            if per_token_logprobs_obj is not None:
                for logprob_dict in per_token_logprobs_obj:
                    token_id, logprob_obj = list(logprob_dict.items())[0]
                    per_token_logprobs.append(logprob_obj.logprob)
            #         per_token_logprobs.append({"token_id": token_id, "logprob": logprob_obj.logprob, "token": logprob_obj.decoded_token, "rank": logprob_obj.rank})
            

            return {
                "response_text": response_text,
                "response_tokens": response_tokens,
                "num_tokens": len(response_tokens),
                "logprobs": cumulative_logprob,
                "per_token_logprobs": per_token_logprobs
            }
        
        return None

    async def generate_batch_async(self, conversations, n_responses_per_conv=4, 
                                   temperature=1.0, max_tokens=1000, logprobs=1, filter_unique=True, schema=None, **kwargs):
        all_tasks = []
        
        # Create tasks for all conversation-response pairs
        for conv_idx, conversation in enumerate(conversations):
            for response_idx in range(n_responses_per_conv):
                task = self._generate_single_with_metadata(
                    conversation, conv_idx, response_idx, temperature, max_tokens, logprobs, schema, **kwargs
                )
                all_tasks.append(task)
        
        # Execute all tasks concurrently
        # print(f"Executing {len(all_tasks)} generation tasks concurrently...")
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Organize results by conversation
        organized_results = [[] for _ in range(len(conversations))]
        
        for result in results:
            if isinstance(result, Exception):
                print(f"Task failed with exception: {result}")
                continue
            
            if result is not None:
                conv_idx = result['conv_idx']
                organized_results[conv_idx].append(result['response'])
        
        # Apply unique filtering if requested
        if filter_unique:
            for conv_idx in range(len(organized_results)):
                unique_responses = []
                unique_responses_set = set()
                for response in organized_results[conv_idx]:
                    if response["response_text"] not in unique_responses_set:
                        unique_responses.append(response)
                        unique_responses_set.add(response["response_text"])
                organized_results[conv_idx] = unique_responses
        
        return organized_results

    async def _generate_single_with_metadata(self, conversation, conv_idx, response_idx, 
                                             temperature, max_tokens, logprobs, schema, **kwargs):
        """Helper method to generate a single response with metadata for organization"""
        try:
            response = await self.generate_single(conversation, temperature, max_tokens, logprobs, schema, **kwargs)
            return {'conv_idx': conv_idx, 'response_idx': response_idx, 'response': response}
        except Exception as e:
            print(f"Error generating response for conv {conv_idx}, response {response_idx}: {e}")
            return None

    async def generate_batch(self, conversation, n_responses=4, temperature=1.0, 
                           max_tokens=1000, logprobs=1, filter_unique=True, schema=None, **kwargs):
        """
        Generate multiple responses for a single conversation (backward compatibility)
        """
        conversations = [conversation]
        results = await self.generate_batch_async(
            conversations, n_responses, temperature, max_tokens, logprobs, filter_unique, schema, **kwargs
        )
        return results[0] if results else []

    def load_model(self, path):
        """Load model configuration and reinitialize"""
        config_path = os.path.join(path, "vllm_async_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            
            model_name = config["model_name"]
            enable_prefix_caching = config.get("enable_prefix_caching", True)
            max_context_length = config.get("max_context_length", 6000)
            
            # Note: We need to properly shutdown the old engine before creating a new one
            self.__init__(model_name=model_name, 
                         enable_prefix_caching=enable_prefix_caching, max_context_length=max_context_length)
            print(f"Async vLLM model reloaded: {model_name}")
        else:
            self.__init__(model_name=path)
            print(f"Async vLLM model loaded from path: {path}")

    async def shutdown(self):
        """Properly shutdown the async engine"""
        if hasattr(self, 'engine') and self.engine:
            try:
                # Try the standard method first
                if hasattr(self.engine, 'shutdown_background_loop'):
                    self.engine.shutdown_background_loop()
                elif hasattr(self.engine, 'shutdown'):
                    shutdown_result = self.engine.shutdown()
                    if shutdown_result is not None and hasattr(shutdown_result, '__await__'):
                        await shutdown_result
                else:
                    # For older versions or different implementations
                    print("Warning: No shutdown method found on engine")
            except Exception as e:
                print(f"Warning: Error during engine shutdown: {e}")

    async def build_tree(self, conversation, degree=2, depth=3, temperature=1.0, max_tokens=1000, logprobs=1, schema=None):
        """Build a tree of responses using uncertain token splitting"""
        tree = []
        todo = [{"subtree_id": f"A{i}" + ("0" * (depth-1)), "split_idx": 0, "parent_sid": "root", "current_depth": 1} for i in range(degree)]
        id2node = {}
        
        # Track active tasks: {task: todo_item}
        active_tasks = {}

        while todo or active_tasks:
            print(f"Todo pending: {len(todo)}, Active tasks: {len(active_tasks)} ({' '.join([t['subtree_id'] for t in todo])})")
            
            # Launch new tasks for all pending todos
            while todo:
                todo_item = todo.pop(0)
                reconstituted_conversation = reconstitute_conversation(conversation, todo_item, id2node, tokenizer=self.tokenizer)
                task = asyncio.create_task(self.generate_single(reconstituted_conversation, temperature=temperature, max_tokens=max_tokens, logprobs=logprobs, schema=schema))
                active_tasks[task] = todo_item
            
            # Check for completed tasks
            if active_tasks:
                done_tasks, pending_tasks = await asyncio.wait(active_tasks.keys(), timeout=0, return_when=asyncio.FIRST_COMPLETED)
                
                # Process completed tasks
                for task in done_tasks:
                    todo_item = active_tasks[task]
                    response = await task  # Get the result
                    response["subtree_id"] = todo_item["subtree_id"]
                    
                    if todo_item["parent_sid"] == "root":
                        merged_response = response
                        merged_response["response_text_illustrated"] = response["response_text"]
                        merged_response["node_start_idx"] = 0
                        merged_response["node_end_idx"] = len(response["response_tokens"])
                    else:
                        merged_response = merge_responses(response, id2node[todo_item["parent_sid"]], split_idx=todo_item["split_idx"], tokenizer=self.tokenizer)
                    
                    tree.append(merged_response)
                    id2node[todo_item["subtree_id"]] = merged_response
                    
                    # Add new todos if not at max depth
                    if todo_item["current_depth"] < depth:
                        split_indeces = find_split_indeces(merged_response, depth-todo_item["current_depth"], min_index=merged_response["node_start_idx"])
                        for idx, split_idx in enumerate(split_indeces):
                            for j in range(degree-1):
                                current_depth = todo_item["current_depth"]+idx+1
                                current_val = int(todo_item["subtree_id"][current_depth])
                                child_subtree_id = todo_item["subtree_id"][:(current_depth)] + str(current_val+j+1) + todo_item["subtree_id"][current_depth+1:]
                                todo.append({"subtree_id": child_subtree_id, "split_idx": split_idx, "parent_sid": todo_item["subtree_id"], "current_depth": current_depth})
                    
                    # Remove completed task from active tasks
                    del active_tasks[task]
                
                # Update active_tasks to only include pending tasks
                new_active_tasks = {}
                for task in pending_tasks:
                    new_active_tasks[task] = active_tasks[task]
                active_tasks = new_active_tasks
            time.sleep(0.1)

        return sorted(tree, key=lambda x: x["subtree_id"])

    def __del__(self):
        # Clean up resources
        if hasattr(self, 'engine'):
            try:
                # Try to shutdown gracefully
                if hasattr(self.engine, 'shutdown_background_loop'):
                    self.engine.shutdown_background_loop()
                elif hasattr(self.engine, 'shutdown'):
                    # Note: Can't use await in __del__, so this is best effort
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Schedule the shutdown for later
                            loop.create_task(self.engine.shutdown())
                    except:
                        pass
            except:
                pass
            del self.engine
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

if __name__ == "__main__":
    async def main():
        # Initialize the async vLLM model
        print("Initializing AsyncVLLMGenerationModel...")
        model = AsyncVLLMGenerationModel(model_name="microsoft/phi-4", enable_prefix_caching=True, max_context_length=6000)

        # Example 1: Regular generation without schema
        print("\n=== Example 1: Regular generation ===")
        conversation = [{"role": "user", "content": "Tell me a 5-word joke about a random country"}]
        
        N_RESPONSES = 1
        print(f"Generating {N_RESPONSES} responses...")
        import time
        start_time = time.time()
        
        results = await model.generate_batch_async(conversations=[conversation], n_responses_per_conv=N_RESPONSES, temperature=0.8, max_tokens=200, filter_unique=True)
        
        end_time = time.time()

        print(results)
        # Process results
        responses = results[0] if results else []
        total_tokens = sum(response["num_tokens"] for response in responses)
        tokens_per_second = total_tokens / (end_time - start_time)
        
        print(f"\nGenerated {len(responses)} unique responses in {end_time - start_time:.2f} seconds")
        print(f"Average time per response: {(end_time - start_time) / len(responses):.3f} seconds")
        print(f"Total tokens: {total_tokens}")
        print(f"Token throughput: {tokens_per_second:.1f} tokens/second")
        
        # Show logprobs for first response
        if responses:
            first_response = responses[0]
            print(f"\nFirst response logprobs:")
            print(f"Cumulative logprob: {first_response.get('cumulative_logprob')}")
            print(f"Per-token logprobs: {first_response.get('per_token_logprobs')}")

        # Example 2: JSON schema generation
        print("\n=== Example 2: JSON schema generation ===")
        
        # Define a JSON schema for a product review
        json_schema = {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                "review_text": {"type": "string"},
                "pros": {"type": "array", "items": {"type": "string"}},
                "cons": {"type": "array", "items": {"type": "string"}},
                "recommended": {"type": "boolean"}
            },
            "required": ["product_name", "rating", "review_text", "recommended"]
        }
        
        schema_conversation = [
            {"role": "user", "content": "Generate a product review for a wireless headphone in JSON format"}
        ]
        
        print("Generating JSON-structured response...")
        start_time = time.time()
        
        schema_result = await model.generate_single(
            conversation=schema_conversation,
            temperature=0.7,
            max_tokens=300,
            schema=json_schema
        )
        
        end_time = time.time()
        
        if schema_result:
            print(f"\nJSON Schema Response:")
            print(f"Response text: {schema_result['response_text']}")
            print(f"Generated in {end_time - start_time:.2f} seconds")
            print(f"Tokens: {schema_result['num_tokens']}")
            
            # Try to parse as JSON to validate structure
            try:
                parsed_json = json.loads(schema_result['response_text'])
                print(f"\nSuccessfully parsed JSON structure:")
                print(json.dumps(parsed_json, indent=2))
            except json.JSONDecodeError as e:
                print(f"\nWarning: Generated text is not valid JSON: {e}")
        else:
            print("No response generated for schema example")
        
        # Shutdown the model
        await model.shutdown()
        print("\nModel shutdown complete.")
    
    # Run the async main function
    asyncio.run(main())
