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


import re

import textworld
from textworld.agents import HumanAgent
from textworld.core import GameState
from textworld.envs.wrappers import Filter
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
from typing import List, Tuple

from ..base.env import BaseEnv


class TextWorldEnvBase(BaseEnv):
    def __init__(self, instance_file: str):
        self.instance_file = instance_file
        self.game_agent = HumanAgent()


    def init_env(self):
        pass


    def _sanitize_command(self, command: str):
        """Fix error of Z-machine: DUMB-FROTZ: unknown escape char: """
        if not isinstance(command, str):
            return ""
        
        # Remove all backslashes - they're rarely valid in text adventure commands
        return command.replace('\\', '').strip()


    def _safe_step(self, command: str):
        """Safely execute a step, falling back to empty command on Unicode errors"""
        command = self._sanitize_command(command)
        try:
            return self.env.step(command)
        except:
            print(f"Game backend error with command '{command}'")
            return self.env.step("")
        
    
    def one_step(self, command: str):
        pass


    def replay(self, commands: List[str]):
        pass

    
    def get_total_rewards(self):
        # For Textworld games, the total reward is typically the max score of the game.
        return 1.0


class TextWorldEnv(TextWorldEnvBase):
    def __init__(self, instance_file: str):
        super().__init__(instance_file)
        self.init_env()


    def init_env(self):
        # Load textworld game:
        textworld_infos = textworld.EnvInfos(
            feedback=True,    # Response from the game after typing a text command.
            description=True, # Text describing the room the player is currently in.
            inventory=True    # Text describing the player's inventory.
        )
        self.env = textworld.start(self.instance_file, request_infos=textworld_infos, wrappers=self.game_agent.wrappers)
        self.game_agent.reset(self.env)
        # Get the initial observation text
        self.init_state = self.format_observation(self.env.reset())


    def format_observation(self, game_state: GameState):
        """
        Get the observation at each step, consisting of `room description`, `game feedback`, `inventory`, and `last action`, according to KG-A2C paper.
        Descriptions:
            - room description: agent's current location
            - game feedback: outputs of game simulator given agent's previous action
            - inventory: agent's inventory list
        """
        room_id = None
        for s in game_state._facts:
            if s.name == "at" and s.arguments[0].name == "P":
                room_id = s.arguments[1].name
                break
        if not room_id or room_id not in game_state.game.infos:
            raise ValueError
        room_desc = f"You are now in the {game_state.game.infos[room_id].name}.\n"

        feedback = self._extract_essential_feedback(game_state.feedback) + '\n'

        inventory = game_state.inventory

        obs = room_desc + feedback + inventory

        return obs


    def one_step(self, command: str):
        game_state, reward, done = self._safe_step(command)
        obs = self.format_observation(game_state)
        return obs, done, reward


    def replay(self, commands: List[str]) -> Tuple[str, bool, float]:
        """
        Reset the game environment, restart game, and replay the sequence of actions.
        Return observation (str), if has won the game (bool), and the reward at the step (float) 
        """
        is_winning = False
        for command in commands:
            game_state, reward, done = self._safe_step(command)
            obs = self.format_observation(game_state)
            if done:
                is_winning = True
                break  # Game completed early
        return obs, is_winning, reward


    def _extract_essential_feedback(self, text):
        """
        Extract essential feedback from TextWorld output.
        This extracts room descriptions and action feedback without duplication.
        """
        result = []
        
        # Extract room descriptions - between room header and prompt
        room_pattern = r'-= (.+?) =-\n([\s\S]*?)(?=\s*>|$)'
        room_matches = re.finditer(room_pattern, text)
        
        for match in room_matches:
            room_desc = match.group(2).strip()
            if room_desc:
                # Split by lines and add non-empty ones
                lines = [line.strip() for line in room_desc.split('\n') if line.strip()]
                result.extend(lines)
        
        # Extract action feedback - lines before a prompt that aren't part of headers
        action_pattern = r'^([^-=>\n][^\n]*?)(?=\n\s*>)'
        action_matches = re.finditer(action_pattern, text, re.MULTILINE)
        
        for match in action_matches:
            action = match.group(1).strip()
            if action and action not in result:
                result.append(action)
        
        # Remove duplicates while preserving order
        return '\n'.join(dict.fromkeys(result))


class AlfWorldEnv(TextWorldEnvBase):
    def __init__(self, instance_file: str):
        super().__init__(instance_file)
        self.init_env()
        

    def init_env(self):
        # Load alfworld game
        self.infos = textworld.EnvInfos(
            score=True,
            max_score=True,
            won=True,
            lost=True,
            feedback=True,
            moves=True,
            extras=["walkthrough", "expert_plan"],
        )
        self.env = textworld.start(
            self.instance_file, self.infos, wrappers=[Filter, AlfredDemangler()]
        )
        self.game_agent.reset(self.env)
        # Get the initial observation text
        obs, info = self.env.reset()
        self.init_state = info["feedback"] + '\n' + "Your admissible actions are: " + ';'.join(info["admissible_commands"])
        # self.init_state = self.env.reset()[1]["feedback"]

    
    def one_step(self, command: str):
        obs, reward, done, extras = self._safe_step(command)
        obs = obs + '\n' + "Your admissible actions are: " + ';'.join(extras["admissible_commands"])
        return obs, done, reward


    def replay(self, commands: List[str]) -> str:
        """
        Reset the game environment, restart game, and replay the sequence of actions.
        This ensures we backtrack correctly to the desired state.
        """
        is_winning = False
        for command in commands:
            obs, reward, done, extras = self._safe_step(command)
            if done:
                is_winning = True
                break  # Game completed early
        return obs, is_winning, reward