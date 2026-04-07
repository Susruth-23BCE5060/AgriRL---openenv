# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Agriculture Environment Implementation.

An agriculture decision environment where the agent selects crops based on
soil, nutrient, climate, and groundwater conditions to maximize reward.
"""

import random
from typing import Dict, Tuple

from models import (
    AgricultureAction,
    AgricultureInfo,
    AgricultureObservation,
    AgricultureState,
    AgricultureTaskConfig,
)

# =========================================================
# Task Definitions
# =========================================================
TASKS: Dict[str, AgricultureTaskConfig] = {
    "crop-selection-easy": AgricultureTaskConfig(
        name="crop-selection-easy",
        max_steps=1,
        description="Choose the most suitable crop for a simple farm state.",
        decision_sequence=["crop"],
    ),
    "farm-planning-medium": AgricultureTaskConfig(
        name="farm-planning-medium",
        max_steps=3,
        description="Plan crop, irrigation, and fertilizer for a farm scenario.",
        decision_sequence=["crop", "irrigation", "fertilizer"],
    ),
    "sustainable-farming-hard": AgricultureTaskConfig(
        name="sustainable-farming-hard",
        max_steps=5,
        description="Build a sustainable farm plan across crop, irrigation, fertilizer, pest control, and strategy.",
        decision_sequence=["crop", "irrigation", "fertilizer", "pest_control", "strategy"],
    ),
}


# =========================================================
# Domain Knowledge
# =========================================================
CROPS = ["rice", "wheat", "maize", "cotton", "millet", "sugarcane", "pulses"]
IRRIGATION = ["drip", "sprinkler", "flood", "rainfed"]
FERTILIZERS = ["nitrogen-rich", "balanced-npk", "organic-compost", "phosphorus-boost"]
PEST_CONTROL = ["none", "integrated-pest-management", "chemical-pesticide", "biological-control"]
STRATEGIES = ["groundwater-conservation", "maximize-yield", "low-cost-farming", "soil-restoration"]

SOIL_CROP_PREF = {
    "loamy": {"wheat": 0.9, "maize": 0.85, "pulses": 0.8, "cotton": 0.7, "rice": 0.65, "millet": 0.75, "sugarcane": 0.7},
    "clay": {"rice": 0.95, "sugarcane": 0.8, "cotton": 0.75, "wheat": 0.7, "maize": 0.65, "millet": 0.55, "pulses": 0.6},
    "sandy": {"millet": 0.9, "pulses": 0.8, "cotton": 0.75, "maize": 0.65, "wheat": 0.55, "rice": 0.35, "sugarcane": 0.4},
    "black": {"cotton": 0.95, "millet": 0.8, "sugarcane": 0.75, "pulses": 0.7, "maize": 0.7, "wheat": 0.65, "rice": 0.55},
    "alluvial": {"rice": 0.9, "wheat": 0.9, "sugarcane": 0.85, "maize": 0.8, "pulses": 0.75, "cotton": 0.7, "millet": 0.65},
}

SEASON_CROP_PREF = {
    "kharif": {"rice": 0.95, "maize": 0.85, "cotton": 0.85, "millet": 0.8, "sugarcane": 0.75, "pulses": 0.7, "wheat": 0.4},
    "rabi": {"wheat": 0.95, "pulses": 0.85, "maize": 0.7, "millet": 0.7, "sugarcane": 0.65, "cotton": 0.45, "rice": 0.5},
    "zaid": {"maize": 0.8, "millet": 0.8, "pulses": 0.75, "sugarcane": 0.7, "rice": 0.6, "cotton": 0.6, "wheat": 0.35},
}

CROP_WATER_NEED = {
    "rice": 0.95,
    "sugarcane": 0.9,
    "cotton": 0.75,
    "maize": 0.65,
    "wheat": 0.6,
    "pulses": 0.4,
    "millet": 0.25,
}

CROP_HEAT_PREF = {
    "rice": 0.7,
    "wheat": 0.4,
    "maize": 0.65,
    "cotton": 0.85,
    "millet": 0.8,
    "sugarcane": 0.75,
    "pulses": 0.55,
}

CROP_NITROGEN_NEED = {
    "rice": 0.75,
    "wheat": 0.7,
    "maize": 0.8,
    "cotton": 0.65,
    "millet": 0.35,
    "sugarcane": 0.85,
    "pulses": 0.25,
}


# =========================================================
# Environment
# =========================================================
class AgricultureEnvironment:
    def __init__(self, task_name: str = "crop-selection-easy", seed: int = 42):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")

        self.task_config = TASKS[task_name]
        self.rng = random.Random(seed)
        self.seed = seed
        self._state: AgricultureState | None = None

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------
    def reset(self) -> AgricultureObservation:
        self._state = self._generate_initial_state()
        return self._state

    def state(self) -> AgricultureState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def step(self, action: AgricultureAction) -> Tuple[AgricultureObservation, float, bool, AgricultureInfo]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        current_step = self._state.step_index
        decision_type = self.task_config.decision_sequence[current_step]

        reward, breakdown, explanation = self._apply_action(decision_type, action.action)
        self._state.step_index += 1

        # state transition after each decision
        self._simulate_state_transition(decision_type)

        done = self._state.step_index >= self._state.max_steps
        final_score = self._compute_final_score() if done else reward
        success = final_score >= self._success_threshold() if done else False

        info = AgricultureInfo(
            current_decision=decision_type,
            reward_breakdown=breakdown,
            score=round(final_score, 4),
            success=success,
            explanation=explanation,
        )

        return self._to_observation(self._state), round(reward, 4), done, info

    # -----------------------------------------------------
    # State Generation
    # -----------------------------------------------------
    def _generate_initial_state(self) -> AgricultureState:
        task_name = self.task_config.name

        if task_name == "crop-selection-easy":
            return AgricultureState(
                task=task_name,
                step_index=0,
                max_steps=self.task_config.max_steps,
                soil_type=self.rng.choice(["loamy", "clay", "alluvial"]),
                nitrogen=round(self.rng.uniform(0.45, 0.85), 2),
                phosphorus=round(self.rng.uniform(0.45, 0.85), 2),
                potassium=round(self.rng.uniform(0.45, 0.85), 2),
                rainfall=round(self.rng.uniform(0.55, 0.95), 2),
                temperature=round(self.rng.uniform(0.45, 0.75), 2),
                groundwater=round(self.rng.uniform(0.55, 0.95), 2),
                pest_risk=round(self.rng.uniform(0.2, 0.5), 2),
                soil_health=round(self.rng.uniform(0.6, 0.9), 2),
                season=self.rng.choice(["kharif", "rabi"]),
            )

        elif task_name == "farm-planning-medium":
            return AgricultureState(
                task=task_name,
                step_index=0,
                max_steps=self.task_config.max_steps,
                soil_type=self.rng.choice(["loamy", "clay", "sandy", "black", "alluvial"]),
                nitrogen=round(self.rng.uniform(0.25, 0.8), 2),
                phosphorus=round(self.rng.uniform(0.25, 0.8), 2),
                potassium=round(self.rng.uniform(0.25, 0.8), 2),
                rainfall=round(self.rng.uniform(0.25, 0.85), 2),
                temperature=round(self.rng.uniform(0.35, 0.85), 2),
                groundwater=round(self.rng.uniform(0.25, 0.85), 2),
                pest_risk=round(self.rng.uniform(0.2, 0.7), 2),
                soil_health=round(self.rng.uniform(0.35, 0.85), 2),
                season=self.rng.choice(["kharif", "rabi", "zaid"]),
            )

        else:  # sustainable-farming-hard
            return AgricultureState(
                task=task_name,
                step_index=0,
                max_steps=self.task_config.max_steps,
                soil_type=self.rng.choice(["loamy", "clay", "sandy", "black", "alluvial"]),
                nitrogen=round(self.rng.uniform(0.1, 0.7), 2),
                phosphorus=round(self.rng.uniform(0.1, 0.7), 2),
                potassium=round(self.rng.uniform(0.1, 0.7), 2),
                rainfall=round(self.rng.uniform(0.1, 0.8), 2),
                temperature=round(self.rng.uniform(0.4, 0.95), 2),
                groundwater=round(self.rng.uniform(0.1, 0.7), 2),
                pest_risk=round(self.rng.uniform(0.35, 0.9), 2),
                soil_health=round(self.rng.uniform(0.2, 0.75), 2),
                season=self.rng.choice(["kharif", "rabi", "zaid"]),
            )

    # -----------------------------------------------------
    # Action Application
    # -----------------------------------------------------
    def _apply_action(self, decision_type: str, action: str) -> Tuple[float, Dict[str, float], str]:
        if decision_type == "crop":
            return self._evaluate_crop(action)
        elif decision_type == "irrigation":
            return self._evaluate_irrigation(action)
        elif decision_type == "fertilizer":
            return self._evaluate_fertilizer(action)
        elif decision_type == "pest_control":
            return self._evaluate_pest_control(action)
        elif decision_type == "strategy":
            return self._evaluate_strategy(action)
        else:
            return 0.0, {"invalid": 0.0}, "Unknown decision type."

    # -----------------------------------------------------
    # Decision Evaluators
    # -----------------------------------------------------
    def _evaluate_crop(self, crop: str) -> Tuple[float, Dict[str, float], str]:
        if crop not in CROPS:
            return 0.0, {"invalid_crop": 0.0}, f"'{crop}' is not a valid crop."

        s = self._state
        assert s is not None

        soil_match = SOIL_CROP_PREF[s.soil_type][crop]
        season_match = SEASON_CROP_PREF[s.season][crop]
        water_match = 1.0 - abs(CROP_WATER_NEED[crop] - s.groundwater)
        climate_match = 1.0 - abs(CROP_HEAT_PREF[crop] - s.temperature)
        nutrient_match = 1.0 - abs(CROP_NITROGEN_NEED[crop] - s.nitrogen)

        score = (
            0.30 * soil_match +
            0.25 * season_match +
            0.20 * water_match +
            0.15 * climate_match +
            0.10 * nutrient_match
        )

        s.chosen_crop = crop

        breakdown = {
            "soil_match": round(soil_match, 4),
            "season_match": round(season_match, 4),
            "water_match": round(water_match, 4),
            "climate_match": round(climate_match, 4),
            "nutrient_match": round(nutrient_match, 4),
        }
        return self._clamp(score), breakdown, f"Crop '{crop}' evaluated."

    def _evaluate_irrigation(self, irrigation: str) -> Tuple[float, Dict[str, float], str]:
        if irrigation not in IRRIGATION:
            return 0.0, {"invalid_irrigation": 0.0}, f"'{irrigation}' is not a valid irrigation strategy."

        s = self._state
        assert s is not None

        crop = s.chosen_crop or "maize"

        # Higher value = more water-intensive irrigation
        irrigation_water_delivery = {
            "drip": 0.30,
            "sprinkler": 0.55,
            "flood": 0.95,
            "rainfed": 0.25,
        }[irrigation]

        water_need = CROP_WATER_NEED[crop]
        crop_fit = 1.0 - abs(irrigation_water_delivery - water_need)

        if s.groundwater < 0.4:
            sustainability_fit = {
                "drip": 0.95,
                "sprinkler": 0.75,
                "rainfed": 0.80 if s.rainfall > 0.55 else 0.45,
                "flood": 0.20,
            }[irrigation]
        else:
            sustainability_fit = {
                "drip": 0.85,
                "sprinkler": 0.80,
                "rainfed": 0.65 if s.rainfall > 0.55 else 0.35,
                "flood": 0.60,
            }[irrigation]

        score = 0.60 * crop_fit + 0.40 * sustainability_fit

        s.chosen_irrigation = irrigation

        breakdown = {
            "crop_fit": round(crop_fit, 4),
            "sustainability_fit": round(sustainability_fit, 4),
        }
        return self._clamp(score), breakdown, f"Irrigation '{irrigation}' evaluated."

    def _evaluate_fertilizer(self, fertilizer: str) -> Tuple[float, Dict[str, float], str]:
        if fertilizer not in FERTILIZERS:
            return 0.0, {"invalid_fertilizer": 0.0}, f"'{fertilizer}' is not a valid fertilizer plan."

        s = self._state
        assert s is not None

        n_def = 1.0 - s.nitrogen
        p_def = 1.0 - s.phosphorus
        k_def = 1.0 - s.potassium

        if fertilizer == "nitrogen-rich":
            correction = n_def
            sustainability = 0.50
            crop_fit = 0.90 if (s.chosen_crop in ["rice", "wheat", "maize", "sugarcane"]) else 0.65

        elif fertilizer == "balanced-npk":
            correction = (n_def + p_def + k_def) / 3
            sustainability = 0.80
            crop_fit = 0.85

        elif fertilizer == "organic-compost":
            correction = (0.6 * n_def + 0.6 * p_def + 0.6 * k_def)
            sustainability = 0.95
            crop_fit = 0.90 if s.soil_health < 0.6 else 0.75

        else:  # phosphorus-boost
            correction = p_def
            sustainability = 0.60
            crop_fit = 0.80

        correction = min(correction, 1.0)
        score = 0.55 * correction + 0.25 * sustainability + 0.20 * crop_fit

        s.chosen_fertilizer = fertilizer

        breakdown = {
            "nutrient_correction": round(correction, 4),
            "soil_sustainability": round(sustainability, 4),
            "crop_fit": round(crop_fit, 4),
        }
        return self._clamp(score), breakdown, f"Fertilizer '{fertilizer}' evaluated."

    def _evaluate_pest_control(self, pest_control: str) -> Tuple[float, Dict[str, float], str]:
        if pest_control not in PEST_CONTROL:
            return 0.0, {"invalid_pest_control": 0.0}, f"'{pest_control}' is not a valid pest control strategy."

        s = self._state
        assert s is not None

        risk = s.pest_risk

        effectiveness = {
            "none": 0.10,
            "integrated-pest-management": 0.80,
            "chemical-pesticide": 0.95,
            "biological-control": 0.65,
        }[pest_control]

        ecological_safety = {
            "none": 1.00,
            "integrated-pest-management": 0.90,
            "chemical-pesticide": 0.35,
            "biological-control": 0.95,
        }[pest_control]

        protection_fit = 1.0 - abs(effectiveness - risk)

        if risk > 0.75 and pest_control == "none":
            protection_fit -= 0.25
        if risk > 0.75 and pest_control == "biological-control":
            protection_fit -= 0.10

        score = 0.65 * protection_fit + 0.35 * ecological_safety

        s.chosen_pest_control = pest_control

        breakdown = {
            "protection_fit": round(self._clamp(protection_fit), 4),
            "ecological_safety": round(ecological_safety, 4),
        }
        return self._clamp(score), breakdown, f"Pest control '{pest_control}' evaluated."

    def _evaluate_strategy(self, strategy: str) -> Tuple[float, Dict[str, float], str]:
        if strategy not in STRATEGIES:
            return 0.0, {"invalid_strategy": 0.0}, f"'{strategy}' is not a valid strategy."

        s = self._state
        assert s is not None

        groundwater_need = 1.0 - s.groundwater
        soil_recovery_need = 1.0 - s.soil_health

        if strategy == "groundwater-conservation":
            fit = 0.7 * groundwater_need + 0.3 * 0.8
            long_term = 0.9

        elif strategy == "maximize-yield":
            fit = 0.78 if (s.chosen_crop in ["rice", "sugarcane", "cotton", "maize", "wheat"]) else 0.55
            long_term = 0.40

        elif strategy == "low-cost-farming":
            fit = 0.65
            long_term = 0.70

        else:  # soil-restoration
            fit = 0.7 * soil_recovery_need + 0.3 * 0.85
            long_term = 0.95

        final = 0.7 * fit + 0.3 * long_term

        s.chosen_strategy = strategy

        breakdown = {
            "strategy_fit": round(fit, 4),
            "long_term_value": round(long_term, 4),
        }
        return self._clamp(final), breakdown, f"Strategy '{strategy}' evaluated."

    # -----------------------------------------------------
    # State Transition
    # -----------------------------------------------------
    def _simulate_state_transition(self, decision_type: str) -> None:
        s = self._state
        assert s is not None

        # Crop affects groundwater and soil health
        if decision_type == "crop" and s.chosen_crop:
            water_need = CROP_WATER_NEED[s.chosen_crop]
            s.groundwater = self._clamp(s.groundwater - 0.15 * water_need)
            s.soil_health = self._clamp(s.soil_health - 0.05 * water_need)

        # Irrigation can preserve or waste water
        if decision_type == "irrigation" and s.chosen_irrigation:
            if s.chosen_irrigation == "drip":
                s.groundwater = self._clamp(s.groundwater + 0.08)
            elif s.chosen_irrigation == "sprinkler":
                s.groundwater = self._clamp(s.groundwater + 0.03)
            elif s.chosen_irrigation == "flood":
                s.groundwater = self._clamp(s.groundwater - 0.08)
            elif s.chosen_irrigation == "rainfed":
                s.groundwater = self._clamp(s.groundwater + 0.02)

        # Fertilizer improves nutrient profile / soil health differently
        if decision_type == "fertilizer" and s.chosen_fertilizer:
            if s.chosen_fertilizer == "nitrogen-rich":
                s.nitrogen = self._clamp(s.nitrogen + 0.25)
                s.soil_health = self._clamp(s.soil_health - 0.03)
            elif s.chosen_fertilizer == "balanced-npk":
                s.nitrogen = self._clamp(s.nitrogen + 0.15)
                s.phosphorus = self._clamp(s.phosphorus + 0.15)
                s.potassium = self._clamp(s.potassium + 0.15)
            elif s.chosen_fertilizer == "organic-compost":
                s.nitrogen = self._clamp(s.nitrogen + 0.10)
                s.phosphorus = self._clamp(s.phosphorus + 0.10)
                s.potassium = self._clamp(s.potassium + 0.10)
                s.soil_health = self._clamp(s.soil_health + 0.10)
            elif s.chosen_fertilizer == "phosphorus-boost":
                s.phosphorus = self._clamp(s.phosphorus + 0.25)

        # Pest control changes future pest pressure / ecological cost
        if decision_type == "pest_control" and s.chosen_pest_control:
            if s.chosen_pest_control == "none":
                s.pest_risk = self._clamp(s.pest_risk + 0.10)
            elif s.chosen_pest_control == "integrated-pest-management":
                s.pest_risk = self._clamp(s.pest_risk - 0.20)
            elif s.chosen_pest_control == "chemical-pesticide":
                s.pest_risk = self._clamp(s.pest_risk - 0.30)
                s.soil_health = self._clamp(s.soil_health - 0.08)
            elif s.chosen_pest_control == "biological-control":
                s.pest_risk = self._clamp(s.pest_risk - 0.18)
                s.soil_health = self._clamp(s.soil_health + 0.03)

        # Strategy affects long-term state
        if decision_type == "strategy" and s.chosen_strategy:
            if s.chosen_strategy == "groundwater-conservation":
                s.groundwater = self._clamp(s.groundwater + 0.12)
            elif s.chosen_strategy == "maximize-yield":
                s.soil_health = self._clamp(s.soil_health - 0.07)
            elif s.chosen_strategy == "low-cost-farming":
                s.soil_health = self._clamp(s.soil_health - 0.02)
            elif s.chosen_strategy == "soil-restoration":
                s.soil_health = self._clamp(s.soil_health + 0.15)

    # -----------------------------------------------------
    # Final Scoring
    # -----------------------------------------------------
    def _compute_final_score(self) -> float:
        s = self._state
        assert s is not None

        crop_score = self._recompute_crop_score()
        irrigation_score = self._recompute_irrigation_score()
        fertilizer_score = self._recompute_fertilizer_score()
        pest_score = self._recompute_pest_score()
        strategy_score = self._recompute_strategy_score()
        consistency_score = self._plan_consistency_score()

        if s.task == "crop-selection-easy":
            return self._clamp(crop_score)

        elif s.task == "farm-planning-medium":
            return self._clamp(
                0.40 * crop_score +
                0.28 * irrigation_score +
                0.22 * fertilizer_score +
                0.10 * consistency_score
            )

        else:  # sustainable-farming-hard
            sustainability_bonus = 0.5 * s.soil_health + 0.5 * s.groundwater
            return self._clamp(
                0.22 * crop_score +
                0.18 * irrigation_score +
                0.18 * fertilizer_score +
                0.14 * pest_score +
                0.10 * strategy_score +
                0.08 * sustainability_bonus +
                0.10 * consistency_score
            )

    def _plan_consistency_score(self) -> float:
        s = self._state
        assert s is not None

        score = 0.50  # neutral base

        crop = s.chosen_crop
        irrigation = s.chosen_irrigation
        fertilizer = s.chosen_fertilizer
        pest = s.chosen_pest_control
        strategy = s.chosen_strategy

        if crop and irrigation:
            if crop == "rice" and irrigation == "flood":
                score += 0.12
            if crop == "rice" and irrigation == "rainfed" and s.rainfall < 0.70:
                score -= 0.18
            if crop in ["millet", "pulses"] and irrigation in ["drip", "rainfed"]:
                score += 0.10
            if crop in ["sugarcane", "rice"] and irrigation == "drip":
                score -= 0.06

        if crop and fertilizer:
            if crop in ["rice", "wheat", "maize", "sugarcane"] and fertilizer == "nitrogen-rich":
                score += 0.08
            if crop == "pulses" and fertilizer == "nitrogen-rich":
                score -= 0.10
            if fertilizer == "organic-compost" and s.soil_health < 0.60:
                score += 0.08

        if pest and strategy:
            if strategy == "soil-restoration" and pest == "biological-control":
                score += 0.10
            if strategy == "groundwater-conservation" and pest == "chemical-pesticide":
                score -= 0.05

        if irrigation and strategy:
            if strategy == "groundwater-conservation" and irrigation == "drip":
                score += 0.14
            if strategy == "groundwater-conservation" and irrigation == "flood":
                score -= 0.18

        if strategy and fertilizer:
            if strategy == "soil-restoration" and fertilizer == "organic-compost":
                score += 0.12
            if strategy == "maximize-yield" and fertilizer in ["balanced-npk", "nitrogen-rich"]:
                score += 0.08

        if pest:
            if s.pest_risk > 0.75 and pest == "none":
                score -= 0.20
            if s.pest_risk > 0.75 and pest == "biological-control":
                score -= 0.08

        return self._clamp(score)

    def _recompute_crop_score(self) -> float:
        s = self._state
        assert s is not None
        if not s.chosen_crop:
            return 0.0

        crop = s.chosen_crop
        return self._clamp(
            0.30 * SOIL_CROP_PREF[s.soil_type][crop] +
            0.25 * SEASON_CROP_PREF[s.season][crop] +
            0.20 * (1.0 - abs(CROP_WATER_NEED[crop] - s.groundwater)) +
            0.15 * (1.0 - abs(CROP_HEAT_PREF[crop] - s.temperature)) +
            0.10 * (1.0 - abs(CROP_NITROGEN_NEED[crop] - s.nitrogen))
        )

    def _recompute_irrigation_score(self) -> float:
        s = self._state
        assert s is not None
        if not s.chosen_irrigation or not s.chosen_crop:
            return 0.0

        irrigation_water_delivery = {
            "drip": 0.30,
            "sprinkler": 0.55,
            "flood": 0.95,
            "rainfed": 0.25,
        }[s.chosen_irrigation]

        water_need = CROP_WATER_NEED[s.chosen_crop]
        crop_fit = 1.0 - abs(irrigation_water_delivery - water_need)

        if s.groundwater < 0.4:
            sustainability_fit = {
                "drip": 0.95,
                "sprinkler": 0.75,
                "rainfed": 0.80 if s.rainfall > 0.55 else 0.45,
                "flood": 0.20,
            }[s.chosen_irrigation]
        else:
            sustainability_fit = {
                "drip": 0.85,
                "sprinkler": 0.80,
                "rainfed": 0.65 if s.rainfall > 0.55 else 0.35,
                "flood": 0.60,
            }[s.chosen_irrigation]

        return self._clamp(0.60 * crop_fit + 0.40 * sustainability_fit)

    def _recompute_fertilizer_score(self) -> float:
        s = self._state
        assert s is not None
        if not s.chosen_fertilizer:
            return 0.0

        if s.chosen_fertilizer == "nitrogen-rich":
            nutrient_fit = 1.0 - abs(0.80 - s.nitrogen)
            sustainability = 0.50
        elif s.chosen_fertilizer == "balanced-npk":
            nutrient_fit = 1.0 - abs(0.75 - ((s.nitrogen + s.phosphorus + s.potassium) / 3))
            sustainability = 0.80
        elif s.chosen_fertilizer == "organic-compost":
            nutrient_fit = 0.75
            sustainability = 0.95 if s.soil_health > 0.55 else 0.85
        else:
            nutrient_fit = 1.0 - abs(0.75 - s.phosphorus)
            sustainability = 0.60

        return self._clamp(0.65 * nutrient_fit + 0.35 * sustainability)

    def _recompute_pest_score(self) -> float:
        s = self._state
        assert s is not None
        if not s.chosen_pest_control:
            return 0.0

        ecological_safety = {
            "none": 1.00,
            "integrated-pest-management": 0.90,
            "chemical-pesticide": 0.35,
            "biological-control": 0.95,
        }[s.chosen_pest_control]

        return self._clamp((1.0 - s.pest_risk) * 0.65 + ecological_safety * 0.35)

    def _recompute_strategy_score(self) -> float:
        s = self._state
        assert s is not None
        if not s.chosen_strategy:
            return 0.0

        if s.chosen_strategy == "groundwater-conservation":
            return self._clamp(0.65 * (1.0 - abs(0.75 - s.groundwater)) + 0.35 * s.soil_health)

        if s.chosen_strategy == "maximize-yield":
            return self._clamp(0.70 * (1.0 - abs(0.75 - s.nitrogen)) + 0.30 * (1.0 - s.soil_health))

        if s.chosen_strategy == "low-cost-farming":
            return self._clamp(0.50 * s.groundwater + 0.50 * s.soil_health)

        return self._clamp(0.70 * s.soil_health + 0.30 * (1.0 - s.pest_risk))

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def _success_threshold(self) -> float:
        if self.task_config.name == "crop-selection-easy":
            return 0.72
        elif self.task_config.name == "farm-planning-medium":
            return 0.74
        return 0.70

    @staticmethod
    def _clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, x))
    
    def _to_observation(self, state: AgricultureState) -> AgricultureObservation:
        return AgricultureObservation(**state.model_dump())