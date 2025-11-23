from typing import Any, Optional
from uuid import uuid4
import ray
from pathlib import Path
from shutil import rmtree
import contextlib
import os
import json
import yaml
import logging
import time
import random
import asyncio

from sweagent.environment.swe_env import SWEEnv # type: ignore
from sweagent.run.common import save_predictions # type: ignore
from sweagent.run.evaluate import evaluate_instance # type: ignore
from verl.utils.rollout_trace import rollout_trace_op
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics, register
from .env_wrapper import batch_instance_from_dict, remove_runtime_root
from .agent import SWEAgent


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@contextlib.contextmanager
def silence_stdout_hard():
    """
    Mute ALL stdout (Python prints and C-level fd=1) inside the block.
    Leaves stderr untouched, so logger output still appears.
    """
    devnull = open(os.devnull, "w")
    saved_fd = os.dup(1)            # save current stdout fd
    try:
        os.dup2(devnull.fileno(), 1)  # point fd=1 to /dev/null
        yield
    finally:
        os.dup2(saved_fd, 1)          # restore stdout
        os.close(saved_fd)
        devnull.close()


@ray.remote(num_cpus=0.01, max_retries=3)
def sweagent_run_remote(
    instance: dict,
    sweagent_config: dict,
    sampling_params: dict[str, Any],
    server_manager,
    tokenizer,
    max_iter: int,
    request_id: str,
    trajs_save_dir: str,
    global_step: int = 0,
    training_phase: str = "train",
    repetition_id: int = 0,
) -> tuple[list[dict[str, str]], float, Optional[str]]:
    """
    Ray remote task that runs a single SWE-agent trajectory in isolation.

    Args:
        instance: SWE-bench instance data containing: 
            - instance_id: unique identifier
            - problem_statement: task description
            - repo: repository URL
            - base_commit: commit hash to checkout
            - test_patch: Test code patch to apply
        sweagent_config: SWE-agent configuration
        sampling_params: vLLM sampling parameters
        server_manager: AsyncLLMServerManager that deals with vllm engine routing
        tokenizer: vLLM tokenizer
        max_iter: Maximum number of agent iterations (0 means no limit)
        request_id: Unique request ID for session affinity
        trajs_save_dir: Base directory to save trajectories
        global_step: Current training iteration
        training_phase: "train" or "eval"
        repetition_id: Trajectory repetition number (for n_samples > 1)

    Returns:
        messages: List of message dicts from SWE-agent
        reward: Float reward based on evaluation
        error: Optional error message if execution failed
    """
      
    # Turn instance dict into a BatchInstance
    batch_instance = batch_instance_from_dict(d=instance)
    instance_id = str(instance.get("instance_id"))

    # Create global directory for each task: <trajs_save_dir>/step_<global_step>/<training_phase>/
    global_path = Path(trajs_save_dir) / f"step_{global_step}" / training_phase
    global_path.mkdir(parents=True, exist_ok=True)

    # Add timestamp to make each runtime root unique
    # NOTE: Ray retries the entire function on failure, so we need a unique path per retry.
    timestamp = int(time.time() * 1000000)  # Microsecond precision
    runtime_root = global_path / f"{instance_id}__run{repetition_id}_{timestamp}"

    if runtime_root.exists():
        logger.info(f"Cleaning up existing runtime root: {runtime_root}")
        try:
            rmtree(runtime_root)
        except Exception as e:
            logger.warning(f"Failed to clean up runtime root: {e}")
            # Continue anyway - Ray will retry if this fails

    runtime_root.mkdir(parents=True, exist_ok=True)

    # Per-trajectory output dir for logs/artifacts (separate from runtime)
    output_dir = global_path / f"{instance_id}_{repetition_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Point SWE-agent at the UNIQUE runtime root
    batch_instance.env.deployment.instance_root = str(runtime_root)
    # Also set conda root inside runtime (for conda env management)
    batch_instance.env.deployment.conda_root = str(runtime_root / ".conda")

    agent = None
    env = None
    result = None
    reward = 0.0
    error = None

    try:
        # Create SWEEnv
        env = SWEEnv.from_config(batch_instance.env)
        
        # Stagger start times to avoid Conda lock contention
        time.sleep(random.uniform(0, 5))
        
        # Start environment
        with silence_stdout_hard():
            env.start()

        # Create SWEAgent
        agent = SWEAgent(
            server_manager=server_manager,
            env=env,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            max_iter=max_iter,
            request_id=request_id,
            agent_config=sweagent_config,
            problem_statement=batch_instance.problem_statement,
            output_dir=output_dir
        )

        with silence_stdout_hard():
            result = agent.run()

    except RuntimeError as e:
        # This includes git clone failures
        # Don't catch it - let Ray retry
        logger.error(f"RuntimeError for {instance_id}: {e}")
        # Clean up env before Ray retries
        if env is not None:
            try:
                env.close()
            except:
                pass
        if runtime_root.exists():
            try:
                rmtree(runtime_root)
            except:
                pass
        # Re-raise so Ray knows to retry
        raise

    except Exception as e:
        # Other exceptions during agent execution
        # These shouldn't be retried - agent failure, not env failure
        logger.error(f"Error during agent execution for {instance_id}: {e}", exc_info=True)
        error = f"{type(e).__name__}: {e}"
        # Don't re-raise - we want to return the error instead
    
    finally:
        # Always close the environment if it was created
        if env is not None:
            try:
                with silence_stdout_hard():
                    env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")

    # Persist outputs or error
    if result is not None and getattr(result, "info", None) is not None:
        save_predictions(output_dir, instance_id, result)
    else:
        (output_dir / "error.txt").write_text(error or "agent returned None")

    # Evaluate if agent completed successfully
    if agent is not None:
        try:
            logger.info(f"Evaluating {instance_id}")
            with silence_stdout_hard():
                eval_summary = evaluate_instance(
                    instance=batch_instance,
                    output_dir=output_dir,
                    timeout=600,
                )
            
            # Write eval summary to output dir
            if eval_summary:
                (output_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2))
            
            # Calculate reward based on pass ratio from eval summary
            if result:
                report = (eval_summary or {}).get("report") or {}
                node = report.get(instance_id) or {}
                pass_ratio = node.get("pass_ratio")
                if pass_ratio is not None:
                    reward = float(pass_ratio)
                    logger.info(f"Instance {instance_id} reward: {reward}")
        
        except Exception as e:
            logger.error(f"Error during evaluation of {instance_id}: {e}", exc_info=True)
            error = f"Evaluation error: {e}"

    messages = agent.messages if agent is not None else []

    # Clean up runtime root after successful completion
    try:
        if runtime_root.exists():
            rmtree(runtime_root)
            logger.info(f"Cleaned up runtime root: {runtime_root}")
    except Exception as e:
        logger.error(f"Failed to clean up runtime root: {e}")

    return messages, reward, error


@register("swe_agent")
class SWEAgentLoop(AgentLoopBase):
    """
    An AgentLoop implementation that runs SWE-agent trajectories using Ray remote tasks.
    This class initializes SWE-agent configuration and manages the execution of
    individual trajectories in isolation via Ray.remote.
    """

    @classmethod
    def init_class(
        cls, 
        config,
        tokenizer, 
        processor, 
        **kwargs
    ):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor

        # Load SWE-agent configuration
        sweagent_config_path = kwargs.get("sweagent_config_path")
        if not sweagent_config_path:
            raise ValueError("sweagent_config_path must be specified in agent_loop_configs.yaml")
        
        logger.info(f"Loading SWE-agent config from: {sweagent_config_path}")
        with open(sweagent_config_path, 'r') as f:
            cls.sweagent_config = yaml.safe_load(f)

        # Agent Loop parameters
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length

        # Get max_iter from agentic config
        cls.max_iter = config.agentic.environment.max_iter
        
        # Trajectory saving (optional)
        cls.trajs_save_dir = kwargs.get("trajs_save_dir")
        if cls.trajs_save_dir:
            os.makedirs(cls.trajs_save_dir, exist_ok=True)
            logger.info(f"Trajectories will be saved to: {cls.trajs_save_dir}")

        # Chat template configuration
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        
        # Compute system prompt token length for later use
        # This helps us separate prompt from response when formatting output
        cls.system_prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}],
            add_generation_prompt=False,
            tokenize=True,
            **cls.apply_chat_template_kwargs
        )

    
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], trajectory_info: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Main entry point for running a single SWE-agent trajectory.

        This method is called by AgentLoopWorker for each sample in a batch.
        It delegates the actual agent execution to a Ray remote task for isolation,
        then formats the results into AgentLoopOutput format.
        
        The workflow:
        1. Extract instance data from kwargs["instance"]
        2. Generate unique request_id for session affinity
        3. Launch Ray task (sweagent_run_remote) with all necessary info
        4. Await results (messages, reward, error)
        5. Format results into AgentLoopOutput

        Args:
            sampling_params: vLLM sampling parameters (temperature, top_p, max_tokens, etc.)
            trajectory_info: Information about the trajectory to run
                - step: Current training step
                - validate: Whether this is an evaluation trajectory
                - sample_index: uid of the sample in the batch
                - rollout_n: trajectory repetition number (for n_samples > 1)
            **kwargs: Additional arguments containing:
                - instance: SWE-Gym/SWE-bench or other instance data

        Returns:
            AgentLoopOutput with:
                - prompt_ids: Initial prompt tokens
                - response_ids: All response tokens (model + observations)
                - response_mask: 1 for model tokens, 0 for observation tokens
                - reward_score: Binary reward (0.0 or 1.0)
                - num_turns: Number of agent turns
                - metrics: Performance metrics
        """
        request_id = uuid4().hex

        # Get instance data from kwargs
        instance = kwargs["instance"]
        
        # Determine output directory
        global_step = trajectory_info.get("step", 0)
        repetition_id = trajectory_info.get("rollout_n", 0)
        training_phase = "eval" if trajectory_info.get("validate", False) else "train"
        uid = trajectory_info.get("sample_index", "unknown")
        # output_base_dir = Path(self.sweagent_traj_dir) / f"step_{global_step}" / training_phase
        
        # Run SWE-agent remotely
        messages, reward, error = await sweagent_run_remote.remote(
            instance=instance,
            sweagent_config=self.sweagent_config,
            sampling_params=sampling_params,
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            max_iter=self.max_iter,
            request_id=request_id,
            trajs_save_dir=self.trajs_save_dir,
            global_step=global_step,
            training_phase=training_phase,
            repetition_id=repetition_id,
        )

        if not messages or error:
            # Return empty trajectory on failure
            if error:
                logger.warning(f"Error in SWE-Agent execution: {error}")
            return self._create_empty_trajectory(kwargs.get("raw_prompt", []), error, uid)
        
        # Process messages to extract prompt_ids and response_ids
        # Assume first 2 messages are system + user (initial prompt)
        initial_messages = messages[:2]
        response_messages = messages[2:]
        
        # Tokenize initial prompt
        initial_input_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                initial_messages,
                add_generation_prompt=False,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        
        # Process response messages
        response_ids = []
        response_mask = []
        
        # Remove trailing user messages (final git diff)
        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        
        if last_idx >= 0:
            response_messages = response_messages[:last_idx + 1]
        
        # Tokenize each response message
        for message in response_messages:
            msg_encoding = await self.loop.run_in_executor(
                None,
                lambda m=message: self.tokenizer.apply_chat_template(
                    [m],
                    add_generation_prompt=False,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            
            response_ids.extend(msg_encoding)
            
            # Mask: 0 for user, 1 for assistant
            if message["role"] == "user":
                response_mask.extend([0] * len(msg_encoding))
            else:  # assistant
                response_mask.extend([1] * len(msg_encoding))
        
        # Truncate to response_length
        response_ids = response_ids[:self.response_length]
        response_mask = response_mask[:self.response_length]

        # Add to extra_fields
        extra_fields = {"error": error}
        extra_fields["uid"] = uid
        
        # Create output
        output = AgentLoopOutput(
            prompt_ids=initial_input_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,  # SWE-agent doesn't provide logprobs
            reward_score=reward,
            final_rewards=reward,
            num_turns=len(messages) // 2,  # Approximate turns
            metrics=AgentLoopMetrics(
                generate_sequences=len(response_messages),
                tool_calls=0,  # SWE-agent uses tools but we don't track separately here
            ),
            extra_fields=extra_fields,
        )
        
        return output
    
    def _create_empty_trajectory(self, raw_prompt: list, error: str, uid: str = None) -> AgentLoopOutput:
        """Create an empty/dummy trajectory for failed cases."""
        failure_message = [{"role": "assistant", "content": f"Failed: {error or 'Unknown error'}"}]
        
        response_ids = self.tokenizer.apply_chat_template(
            failure_message,
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        
        prompt_ids = self.tokenizer.apply_chat_template(
            raw_prompt,
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        
        response_mask = [1] * len(response_ids)
        
        extra_fields = {"error": error, "uid": uid}

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            reward_score=0.0,
            num_turns=1,
            metrics=AgentLoopMetrics(generate_sequences=0, tool_calls=0),
            extra_fields=extra_fields,
        )
