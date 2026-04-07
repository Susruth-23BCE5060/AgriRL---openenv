# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Agriculture Environment Client

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import AgricultureAction


class AgricultureEnvClient(EnvClient):
    """
    Thin client wrapper around OpenEnv server for the agriculture environment.
    """

    def parse_action(self, raw_action: str) -> AgricultureAction:
        """
        Convert model / agent text output into a typed AgricultureAction.
        """
        return AgricultureAction(action=raw_action.strip().lower())

    def format_state_for_prompt(self, state: State) -> str:
        """
        Converts environment state into a readable prompt for LLM / debugging baseline.
        """
        data: Dict[str, Any] = state.data if hasattr(state, "data") else dict(state)

        task = data.get("task", "unknown")
        step_index = data.get("step_index", 0)
        max_steps = data.get("max_steps", 1)

        decision_sequence = {
            "crop-selection-easy": ["crop"],
            "farm-planning-medium": ["crop", "irrigation", "fertilizer"],
            "sustainable-farming-hard": ["crop", "irrigation", "fertilizer", "pest_control", "strategy"],
        }

        current_decision = "unknown"
        if task in decision_sequence and step_index < len(decision_sequence[task]):
            current_decision = decision_sequence[task][step_index]

        prompt = f"""
You are an agriculture planning agent.

Task: {task}
Current Step: {step_index + 1}/{max_steps}
Decision Required: {current_decision}

Farm State:
- Soil Type: {data.get("soil_type")}
- Nitrogen: {data.get("nitrogen")}
- Phosphorus: {data.get("phosphorus")}
- Potassium: {data.get("potassium")}
- Rainfall: {data.get("rainfall")}
- Temperature: {data.get("temperature")}
- Groundwater: {data.get("groundwater")}
- Pest Risk: {data.get("pest_risk")}
- Soil Health: {data.get("soil_health")}
- Season: {data.get("season")}

Choices already made:
- Crop: {data.get("chosen_crop")}
- Irrigation: {data.get("chosen_irrigation")}
- Fertilizer: {data.get("chosen_fertilizer")}
- Pest Control: {data.get("chosen_pest_control")}
- Strategy: {data.get("chosen_strategy")}

Respond with ONLY the best next action.
No explanation.
""".strip()

        return prompt

    def extract_score(self, result: StepResult) -> float:
        """
        Extract normalized score from environment step result.
        """
        try:
            return float(result.info.get("score", result.reward))
        except Exception:
            return float(result.reward)