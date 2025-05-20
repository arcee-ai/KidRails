#! /usr/bin/env python3
import asyncio
import json
import os
import time
import random
import traceback
from typing import Dict, List, Deque, Tuple, Optional, Any
from collections import deque

# Third-party libraries
from openai import AsyncOpenAI

##############################################################################
#                          CONFIGURATION                                       #
##############################################################################

# Rate limits - adjust these based on your API tier
MAX_REQUESTS_PER_MINUTE = 3000     # Conservative default
MAX_TOKENS_PER_MINUTE = 1000000    # Conservative default
MAX_CONCURRENCY = 100              # Reduced for stability

# Processing settings
RETRY_ATTEMPTS = 3
INITIAL_RETRY_DELAY = 1.0
PROCESSING_CHUNK_SIZE = 50         # Reduced chunk size for better memory management
VERBOSE_LOGGING = True             # Enable detailed logging by default

# Progress tracking
PROGRESS_UPDATE_FREQUENCY = 10     # Show progress every N items
SKIP_LOG_FREQUENCY = 100          # Log skipped items every N items

class OptimizedRateLimiter:
    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_times: Deque[float] = deque()
        self.current_requests = 0
        self.token_times: Deque[Tuple[float, int]] = deque()
        self.current_tokens = 0
        self.lock = asyncio.Lock()
        self.last_prune = time.monotonic()

    async def _prune_old(self, now: float):
        """Remove entries older than 60 seconds."""
        while self.request_times and self.request_times[0] < now - 60:
            self.request_times.popleft()
            self.current_requests -= 1
        while self.token_times and self.token_times[0][0] < now - 60:
            self.current_tokens -= self.token_times.popleft()[1]

    async def acquire(self, tokens_needed: int):
        """Acquire permission to make an API request."""
        while True:
            async with self.lock:
                now = time.monotonic()
                if now - self.last_prune >= 0.001:
                    await self._prune_old(now)
                    self.last_prune = now
                if (self.current_requests < self.rpm_limit and
                    self.current_tokens + tokens_needed <= self.tpm_limit):
                    self.request_times.append(now)
                    self.current_requests += 1
                    self.token_times.append((now, tokens_needed))
                    self.current_tokens += tokens_needed
                    return
            await asyncio.sleep(0.001)

def load_seed_data(safe_file: str, unsafe_file: str) -> Tuple[List[Dict], List[Dict]]:
    """Load the seed data from safe and unsafe response files."""
    safe_responses = []
    unsafe_responses = []
    
    try:
        with open(safe_file, 'r') as f:
            safe_responses = json.load(f)
        with open(unsafe_file, 'r') as f:
            unsafe_responses = json.load(f)
    except Exception as e:
        print(f"Error loading seed data: {e}")
        raise
        
    return safe_responses, unsafe_responses

async def _generate_variant_response_with_llm(
    client: AsyncOpenAI,
    original_response: str,
    is_safe: bool,
    rate_limiter: OptimizedRateLimiter
) -> str:
    """Generate a variant response using the LLM."""
    prompt_template = """You are an AI assistant helping to generate variant responses for a dataset. 
Given an original response, create a new response that maintains the same intent and safety characteristics but uses different wording.

Original Response:
{original_response}

Type: {response_type}

Requirements:
1. Maintain the same tone and safety level as the original
2. Use different phrasing while keeping the core message
3. Keep approximately the same length
4. For unsafe queries, maintain the pattern of redirecting to safe topics
5. For safe queries, maintain the educational and engaging style

Generate a new response that follows these requirements.
"""

    response_type = "SAFE" if is_safe else "UNSAFE (requires redirection)"
    prompt = prompt_template.format(
        original_response=original_response,
        response_type=response_type
    )

    try:
        # Estimate tokens for rate limiting
        estimated_tokens = len(prompt.split()) + 100  # rough estimate
        await rate_limiter.acquire(estimated_tokens)

        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Moderate temperature for creativity while maintaining coherence
            min_p=0.1,       # Filter out low-probability tokens
            max_tokens=800,  # Adjust based on your needs
            presence_penalty=0.6,  # Encourage different wording
            frequency_penalty=0.6,  # Discourage repetition
            timeout=30.0
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        return original_response  # Fallback to original if generation fails

    except Exception as e:
        print(f"Error generating variant response: {e}")
        return original_response  # Fallback to original if there's an error

async def generate_new_example(
    client: AsyncOpenAI,
    safe_examples: List[Dict],
    unsafe_examples: List[Dict],
    rate_limiter: OptimizedRateLimiter
) -> Dict:
    """Generate a new example by combining and modifying seed data."""
    # Randomly choose whether to base this on a safe or unsafe example
    is_safe = random.choice([True, False])
    base_examples = safe_examples if is_safe else unsafe_examples
    
    # Select a random example to use as a base
    base_example = random.choice(base_examples)
    
    # Get the original response
    original_response = base_example["response"]["choices"][0]["message"]["content"]
    
    # Generate variant response using LLM
    variant_response = await _generate_variant_response_with_llm(
        client,
        original_response,
        is_safe,
        rate_limiter
    )
    
    # Create a new example with modified content
    new_example = {
        "age": base_example["age"],
        "system_prompt": base_example["system_prompt"],
        "question": _modify_question(base_example["question"]),
        "expected_output": original_response,
        "actual_output": variant_response
    }
    
    return new_example

def _modify_question(question: str) -> str:
    """Create a variant of the original question."""
    # This is a placeholder - in practice, you'd want more sophisticated text manipulation
    modifiers = ["Can you tell me", "I want to know", "Please explain", "Help me understand"]
    return f"{random.choice(modifiers)} {question.lower()}"

async def generate_examples_batch(
    client: AsyncOpenAI,
    safe_examples: List[Dict],
    unsafe_examples: List[Dict],
    batch_size: int,
    rate_limiter: OptimizedRateLimiter
) -> List[Dict]:
    """Generate a batch of new examples."""
    examples = []
    for _ in range(batch_size):
        example = await generate_new_example(
            client,
            safe_examples,
            unsafe_examples,
            rate_limiter
        )
        examples.append(example)
    return examples

async def main():
    """Main execution function."""
    print("Starting example generation script")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
        
    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=30.0
    )
    
    # Load seed data
    safe_file = "safe_api_responses.json"
    unsafe_file = "unsafe_api_responses.json"
    
    try:
        safe_examples, unsafe_examples = load_seed_data(safe_file, unsafe_file)
        print(f"Loaded {len(safe_examples)} safe examples and {len(unsafe_examples)} unsafe examples")
        
        # Initialize rate limiter
        rate_limiter = OptimizedRateLimiter(MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE)
        
        # Generate examples in batches
        total_examples_to_generate = 1000  # Adjust as needed
        examples_generated = 0
        output_file = "generated_examples.jsonl"
        
        while examples_generated < total_examples_to_generate:
            batch_size = min(PROCESSING_CHUNK_SIZE, total_examples_to_generate - examples_generated)
            print(f"\nGenerating batch of {batch_size} examples...")
            
            new_examples = await generate_examples_batch(
                client,
                safe_examples,
                unsafe_examples,
                batch_size,
                rate_limiter
            )
            
            # Save the batch
            with open(output_file, "a") as f:
                for example in new_examples:
                    json.dump(example, f)
                    f.write('\n')
            
            examples_generated += batch_size
            print(f"Progress: {examples_generated}/{total_examples_to_generate} examples generated")
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        print(f"\nGeneration complete! {examples_generated} examples generated and saved to {output_file}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 