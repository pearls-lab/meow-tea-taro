import asyncio
import logging
import os
from typing import Any, Optional

from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.exceptions import ContextWindowExceededError, CostLimitExceededError, InstanceCallLimitExceededError
from tenacity import RetryError

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SWEAgentModelWrapper:
    """
    Wrapper for SWE-agent's DefaultAgent.model to integrate with VeRL's server_manager for Ray RPC-based function calls, instead of LiteLLM HTTP API (default for SWE-agent).
    We implement the query() method that DefaultAgent.model expects, but internally call agent_loop's server_manager.generate().

    Args:
        server_manager: VeRL's AgentServerManager instance for async generation calls.
        tokenizer
        request_id: str, unique ID for this agent session (used for session affinity in VeRL)
        sampling_params: dict, from VeRL's config
        per_instance_call_limit: int, max number of calls allowed per instance. Same as max_iter. (0 = no limit)
        per_instance_cost_limit: float, max cost allowed per instance (0.0 = no limit)
    """
    def __init__(
        self, 
        server_manager,
        tokenizer,
        request_id: str, 
        sampling_params: dict,
        per_instance_call_limit: int,  # 0 means no limit. Same as max_iter.
        per_instance_cost_limit: float = 0.0,  # 0.0 means no limit. We ignore it as we are using open-source models.
    ):
        self.server_manager = server_manager
        self.tokenizer = tokenizer  
        self.request_id = request_id
        self.sampling_params = sampling_params
        self._per_instance_call_limit = per_instance_call_limit
        self._per_instance_cost_limit = per_instance_cost_limit

        # Mock stats object that SWE-Agent expects
        class Stats:
            def __init__(self):
                self.completion_tokens = 0
                self.prompt_tokens = 0
                self.total_cost = 0.0
                self.api_calls = 0
                # Mock cost per token
                # For reference: GPT-4 is ~$0.03/1K prompt tokens, $0.06/1K completion tokens
                self.prompt_cost_per_token = 0.00003
                self.completion_cost_per_token = 0.00006
            
            def update_cost(self, prompt_tokens: int, completion_tokens: int):
                """Update the total cost based on token usage"""
                prompt_cost = prompt_tokens * self.prompt_cost_per_token
                completion_cost = completion_tokens * self.completion_cost_per_token
                self.total_cost += prompt_cost + completion_cost
                return prompt_cost, completion_cost
            
            def model_dump(self):
                return {
                    'completion_tokens': self.completion_tokens,
                    'prompt_tokens': self.prompt_tokens,
                    'total_cost': self.total_cost,
                    'api_calls': self.api_calls,
                }
            
        self.stats = Stats()


    def query(self, history: list[dict[str, str]]) -> dict[str, Any]:
        """
        Wrap DefaultAgent.model.query to call VeRL's async server_manager.generate().
        The query() method is expected to be synchronous, so we need to run the async code in a thread-safe way.

        Args:
            history: Conversation history as list of dicts with 'role' and 'content' keys

        Returns:
            Dict with 'message' key containing the generated text
        """
        # SWE-agent calls this from synchronous code, but server_manager.generate is async
        try:
            # Try to get the running event loop (if any)
            loop = asyncio.get_running_loop()
            logger.debug("[DEBUG] Using existing event loop")
            # Run the async query in the existing loop
            future = asyncio.run_coroutine_threadsafe(
                self._async_query(history),
                loop
            )
            result = future.result()
            return result
        except RuntimeError:
            # No running event loop, create a new one
            logger.debug("[DEBUG] Creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Run the async query in the new loop
                result = loop.run_until_complete(self._async_query(history))
                return result
            finally:
                loop.close()


    async def _async_query(self, history: list[dict[str, str]]) -> dict[str, Any]:
        """
        Internal async method to perform the query using VeRL's server_manager.

        This is the KEY integration point with VeRL!
        Instead of making HTTP requests to LiteLLM, we:
        1. Tokenize messages to get prompt_ids
        2. Call server_manager.generate() with request_id
        3. Decode output tokens back to text
        
        The request_id ensures session affinity:
        - Same request_id always routes to same vLLM server
        - Enables prefix caching for multi-turn efficiency
        - First turn: cache miss, process full prompt
        - Subsequent turns: cache hit, only process new tokens
        
        Args:
            history: Conversation history as list of dicts with 'role' and 'content' keys

        Returns:
            Dict with 'message' key containing the generated text
        """
        # Increment api_calls before the call
        self.stats.api_calls += 1

        # Check call limit before making the call
        if self._per_instance_call_limit > 0:
            if self.stats.api_calls > self._per_instance_call_limit:
                logger.warning(
                    f"[TURN {self.stats.api_calls}] Call limit EXCEEDED: "
                    f"{self.stats.api_calls} > {self._per_instance_call_limit}"
                )
                raise InstanceCallLimitExceededError(
                    f"Per instance call limit exceeded: {self.stats.api_calls} > {self._per_instance_call_limit}"
                )
        
        prompt_ids = self.tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            tokenize=True,
        )
        prompt_length = len(prompt_ids)
        
        # VeRL's server calculates max_tokens = model_max_length - prompt_length internally
        # So we should NOT include max_tokens in sampling_params
        sampling_params_for_server = {
            k: v for k, v in self.sampling_params.items() 
            if k != 'max_tokens'
        }
        
        try:
            # Make the async call to VeRL's server_manager
            output = await self.server_manager.generate(
                request_id=self.request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params_for_server,
                image_data=None,
            )
        except ValueError as e:
            # ERROR scenario: VeRL server raises ValueError when max_tokens would be negative
            # Convert this to ContextWindowExceededError so SWE-agent's error handling works
            if "max_tokens must be at least 1" in str(e):
                logger.warning(
                    f"[TURN {self.stats.api_calls}] Context window exceeded: prompt length {prompt_length} tokens. "
                    f"Raising ContextWindowExceededError for SWE-agent to handle."
                )
                raise ContextWindowExceededError(
                    f"Prompt length {prompt_length} exceeds model's context window"
                ) from e
            # Re-raise other ValueErrors
            raise
        
        except Exception as e:
            # ERROR scenario: Handle Ray/network errors
            # Log the original error for debugging
            logger.warning(f"[TURN {self.stats.api_calls}] Error during VeRL generation: {type(e).__name__}: {e}")
            
            # Convert certain errors to what SWE-agent expects
            error_msg = str(e).lower()
            error_type = type(e).__name__.lower()
            
            # Network/connection/Ray errors -> RetryError equivalent
            if any(keyword in error_msg or keyword in error_type for keyword in [
                'connection', 'timeout', 'network', 'ray', 'rpc', 
                'actor', 'worker', 'unavailable', 'failed'
            ]):
                logger.warning(
                    f"[TURN {self.stats.api_calls}] Network/Ray error detected, converting to RetryError"
                )
                raise RetryError(f"VeRL server error: {e}") from e
            
            # Re-raise as-is for other errors (SWE-agent will catch as generic Exception)
            raise
        
        response_ids = output.token_ids
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Update stats with cost calculation
        prompt_tokens_added = len(prompt_ids)
        completion_tokens_added = len(output.token_ids)
        
        self.stats.completion_tokens += completion_tokens_added
        self.stats.prompt_tokens += prompt_tokens_added
        prompt_cost, completion_cost = self.stats.update_cost(prompt_tokens_added, completion_tokens_added)
        
        # Print detailed cost breakdown for this turn
        logger.info(
            f"[TURN {self.stats.api_calls}] === COST BREAKDOWN ===\n"
            f"  Prompt tokens: {prompt_tokens_added:,} @ ${self.stats.prompt_cost_per_token*1000:.5f}/1K = ${prompt_cost:.6f}\n"
            f"  Completion tokens: {completion_tokens_added:,} @ ${self.stats.completion_cost_per_token*1000:.5f}/1K = ${completion_cost:.6f}\n"
            f"  Turn cost: ${prompt_cost + completion_cost:.6f}\n"
            f"  === CUMULATIVE STATS ===\n"
            f"  Total prompt tokens: {self.stats.prompt_tokens:,}\n"
            f"  Total completion tokens: {self.stats.completion_tokens:,}\n"
            f"  Total cost so far: ${self.stats.total_cost:.6f}\n"
            f"  Cost limit: ${self._per_instance_cost_limit:.6f} {'(NO LIMIT)' if self._per_instance_cost_limit == 0 else ''}"
        )

        # ERROR scenario: Check cost limit if configured
        if self._per_instance_cost_limit > 0:
            if self.stats.total_cost > self._per_instance_cost_limit:
                logger.warning(
                    f"[TURN {self.stats.api_calls}] Cost limit EXCEEDED: "
                    f"${self.stats.total_cost:.6f} > ${self._per_instance_cost_limit:.6f}"
                )
                raise CostLimitExceededError(
                    f"Cost limit exceeded: ${self.stats.total_cost:.6f} > ${self._per_instance_cost_limit:.6f}"
                )
        
        return {"message": response_text}
    

class SWEAgent:
    """
    An instance of SWE-agent's DefaultAgent, but with the model replaced by our SWEAgentModelWrapper.
    """
    def __init__(
        self,
        problem_statement: TextProblemStatement, # initial prompt
        env: SWEEnv,
        server_manager,
        tokenizer,
        sampling_params: dict,
        max_iter: int,
        request_id: str,
        agent_config: dict, # An instance of DefaultAgentConfig
        output_dir, # Path
    ):  
        self.env = env
        self.problem_statement = problem_statement
        self.output_dir = output_dir
        
        sweagent_model_wrapper = SWEAgentModelWrapper(
            server_manager=server_manager,
            tokenizer=tokenizer,
            request_id=request_id,
            sampling_params=sampling_params,
            per_instance_call_limit=max_iter,
        )
        # Initialize DefaultAgent
        # NOTE: This will create a model based on agent_config.model,
        # but we immediately replace it with our wrapper
        self._agent = DefaultAgent.from_config(DefaultAgentConfig.model_validate(agent_config.get("agent", {})))
        
        # Replace the model with our VeRL wrapper
        self._agent.model = sweagent_model_wrapper

        logger.info(f"Initialized SWEAgent with request_id: {request_id}")


    def run(self) -> Any:
        """Run the agent's main loop."""
        return self._agent.run(
            env=self.env,
            problem_statement=self.problem_statement,
            output_dir=self.output_dir
        )

    @property
    def trajectory(self):
        """Access the underlying agent's trajectory."""
        return self._agent.trajectory

    @property
    def info(self):
        """Access the underlying agent's info dict."""
        return self._agent.info

    @property
    def messages(self):
        """Access the underlying agent's messages."""
        return self._agent.messages

