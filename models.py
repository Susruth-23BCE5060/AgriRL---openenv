# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Agriculture Environment.

The agriculture environment is a simple test environment that echoes back messages.
"""

# from openenv.core.env_server.types import Action, Observation
# from pydantic import Field


# class AgricultureAction(Action):
#     """Action for the Agriculture environment - just a message to echo."""

#     message: str = Field(..., description="Message to echo back")


# class AgricultureObservation(Observation):
#     """Observation from the Agriculture environment - the echoed message."""

#     echoed_message: str = Field(default="", description="The echoed message")
#     message_length: int = Field(default=0, description="Length of the echoed message")


from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:  # pragma: no cover
    from pydantic import BaseModel as Action, BaseModel as Observation, BaseModel as State


# -----------------------------
# Core Environment State
# -----------------------------
class AgricultureState(State):
    task: Literal[
        "crop-selection-easy",
        "farm-planning-medium",
        "sustainable-farming-hard",
    ]

    step_index: int = 0
    max_steps: int = 1

    # Farm / environmental features
    soil_type: Literal["loamy", "clay", "sandy", "black", "alluvial"]
    nitrogen: float = Field(..., ge=0.0, le=1.0)
    phosphorus: float = Field(..., ge=0.0, le=1.0)
    potassium: float = Field(..., ge=0.0, le=1.0)
    rainfall: float = Field(..., ge=0.0, le=1.0)
    temperature: float = Field(..., ge=0.0, le=1.0)
    groundwater: float = Field(..., ge=0.0, le=1.0)
    pest_risk: float = Field(..., ge=0.0, le=1.0)
    soil_health: float = Field(..., ge=0.0, le=1.0)

    season: Literal["kharif", "rabi", "zaid"]

    # Decisions taken so far
    chosen_crop: Optional[
        Literal["rice", "wheat", "maize", "cotton", "millet", "sugarcane", "pulses"]
    ] = None

    chosen_irrigation: Optional[
        Literal["drip", "sprinkler", "flood", "rainfed"]
    ] = None

    chosen_fertilizer: Optional[
        Literal["nitrogen-rich", "balanced-npk", "organic-compost", "phosphorus-boost"]
    ] = None

    chosen_pest_control: Optional[
        Literal["none", "integrated-pest-management", "chemical-pesticide", "biological-control"]
    ] = None

    chosen_strategy: Optional[
        Literal["groundwater-conservation", "maximize-yield", "low-cost-farming", "soil-restoration"]
    ] = None


# -----------------------------
# Observation Model
# -----------------------------
class AgricultureObservation(Observation):
    task: str
    step_index: int
    max_steps: int

    soil_type: str
    nitrogen: float
    phosphorus: float
    potassium: float

    rainfall: float
    temperature: float
    groundwater: float

    pest_risk: float
    soil_health: float
    season: str

    reward: float
    done: bool
    info: Optional[dict]


# -----------------------------
# Action Model
# -----------------------------
class AgricultureAction(Action):
    action: str


# -----------------------------
# Step Metadata
# -----------------------------
class AgricultureInfo(BaseModel):
    current_decision: str
    reward_breakdown: Dict[str, float]
    score: float
    success: bool
    explanation: str


# -----------------------------
# Task Metadata
# -----------------------------
class AgricultureTaskConfig(BaseModel):
    name: Literal[
        "crop-selection-easy",
        "farm-planning-medium",
        "sustainable-farming-hard",
    ]
    max_steps: int
    description: str
    decision_sequence: List[str]
