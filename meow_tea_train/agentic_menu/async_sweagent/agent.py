import asyncio
import logging
import os
from typing import Any, Optional

from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.agent.problem_statement import TextProblemStatement

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SWEAgentModelWrapper:
    def __init__(
        self, 
        server_manager,
        tokenizer,
        request_id: str, 
        sampling_params: dict
    ):
        self.server_manager = server_manager
        self.tokenizer = tokenizer  
        self.request_id = request_id
        self.sampling_params = sampling_params

        # Mock stats object that SWE-Agent expects
        self.stats = type('obj', (object,), {
            'model_dump': lambda: {},
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_cost': 0.0,
        })()
        
        logger.info(f"Initialized SWEAgentModelWrapper with request_id: {request_id}")


    def query(self, history: list[dict[str, str]]) -> dict[str, Any]:
        """
        Query the model with conversation history.
        
        This is the method that SWE-Agent's DefaultAgent calls (line 1144).
        We implement it to use VeRL's AsyncLLMServerManager instead of LiteLLM.
        
        Args:
            history: List of message dicts with 'role' and 'content' keys
                    e.g., [{"role": "system", "content": "..."}, ...]
        
        Returns:
            Dict with 'message' key containing the model's response text
            Format: {"message": "generated text"}
        """
        # SWE-agent calls this from synchronous code, but server_manager.generate is async
        # We need to run it in an event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._async_query(history))
            loop.close()
            return result
        else:
            # Event loop already running, use it
            return asyncio.run_coroutine_threadsafe(
                self._async_query(history),
                loop
            ).result()


    async def _async_query(self, history: list[dict[str, str]]) -> dict[str, Any]:
        """
        Async implementation of query.
        
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
            history: Conversation history
        
        Returns:
            Dict with 'message' key containing the generated text
        """
        prompt_ids = self.tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        output = await self.server_manager.generate(
            request_id=self.request_id,  # enable session affinity
            prompt_ids=prompt_ids,
            sampling_params=self.sampling_params,
            image_data=None, # No image data for SWE-agent
        )
        response_ids = output.token_ids
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Update stats (for SWE-agent's tracking)
        self.stats.completion_tokens += len(output.token_ids)
        self.stats.prompt_tokens += len(prompt_ids)
        
        # Return in the format SWE-Agent expects
        return {"message": response_text}
    

class SWEAgent:
    def __init__(
        self,
        problem_statement: TextProblemStatement, # initial prompt
        env: SWEEnv,
        server_manager,  # an instance of AsyncLLMServerManager
        tokenizer,
        sampling_params: dict,
        request_id: str,
        agent_config: dict, # An instance of DefaultAgentConfig
        output_dir, # Path
    ):  
        self.env = env
        self.problem_stratement = problem_statement
        self.output_dir = output_dir
        
        sweagent_model_wrapper = SWEAgentModelWrapper(
            server_manager=server_manager,
            tokenizer=tokenizer,
            request_id=request_id,
            sampling_params=sampling_params
        )
        # Initialize DefaultAgent
        # Note: This will create a model based on agent_config.model,
        # but we immediately replace it with our wrapper
        # self._agent = DefaultAgent(agent_config)
        self._agent = DefaultAgent.from_config(DefaultAgentConfig.model_validate(agent_config.get("agent", {})))
        
        # Replace the model with our VeRL wrapper
        # This is the KEY integration point - everything else stays the same!
        self._agent.model = sweagent_model_wrapper

        logger.info(f"Initialized SWEAgent with request_id: {request_id}")


    def run(self) -> Any:
        return self._agent.run(
            env=self.env,
            problem_statement=self.problem_stratement,
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

