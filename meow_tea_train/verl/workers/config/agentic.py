# Copyright 2025 Ruiyi Wang, PEARLS Lab, UC San Diego
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

__all__ = [
    "AgenticEnvironmentConfig",
    "AgenticAgentConfig",
    "AgenticRewardConfig",
    "AgenticAgentLoopConfig",
]

# NOTE from meow-tea: These are basic configurations for agentic environment, agent and reward.
# You can extend these configs or create your own config class to add more parameters as needed.
@dataclass
class AgenticEnvironmentConfig(BaseConfig):
    name: str = None
    is_multiturn: bool = True
    is_async: bool = False
    max_iter: int = 10
    n_traj: int = 1

    def __post_init__(self):
        """Validate the environment config"""
        assert self.name in ["textworld", "alfworld", "swegym", "custom"], (
            "name must be one of ['textworld', 'alfworld', 'swegym', 'custom']"
        )

@dataclass
class AgenticAgentConfig(BaseConfig):
    use_think: bool = False
    use_tool: bool = False
    use_memory: bool = False


@dataclass
class AgenticRewardConfig(BaseConfig):
    density: str = "single"
    type: str = "verified"

    def __post_init__(self):
        """Validate the reward config"""
        assert self.density in ["single", "dense"], "density must be one of ['single', 'dense']"
        assert self.type in ["verified", "learned"], "type must be one of ['verified', 'learned']"


@dataclass
class SWEAgentKwargs:
    """Configuration for SWE-agent specific parameters"""
    trajs_save_dir: Optional[str] = None
    sweagent_config_path: Optional[str] = None


@dataclass
class AgenticAgentLoopConfig(BaseConfig):
    type: Optional[str] = None
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type == "async_software":
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(self, 'kwargs', SWEAgentKwargs(**self.kwargs))
        elif self.type is None:
            pass
        else:
            raise ValueError(f"Unsupported agent loop type: {self.type}")
