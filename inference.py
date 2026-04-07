"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment if using from_docker_image()

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import argparse
import random

from server.agriculture_environment import AgricultureEnvironment
from models import AgricultureAction


# =========================================================
# Action Spaces
# =========================================================
CROPS = ["rice", "wheat", "maize", "cotton", "millet", "sugarcane", "pulses"]
IRRIGATION = ["drip", "sprinkler", "flood", "rainfed"]
FERTILIZERS = ["nitrogen-rich", "balanced-npk", "organic-compost", "phosphorus-boost"]
PEST_CONTROL = ["none", "integrated-pest-management", "chemical-pesticide", "biological-control"]
STRATEGIES = ["groundwater-conservation", "maximize-yield", "low-cost-farming", "soil-restoration"]


# =========================================================
# Heuristic Baseline Policy
# =========================================================
def choose_action_heuristic(state) -> str:
    task = state.task
    step = state.step_index

    decision_sequence = {
        "crop-selection-easy": ["crop"],
        "farm-planning-medium": ["crop", "irrigation", "fertilizer"],
        "sustainable-farming-hard": ["crop", "irrigation", "fertilizer", "pest_control", "strategy"],
    }

    decision_type = decision_sequence[task][step]

    if decision_type == "crop":
        if state.soil_type == "clay" and state.groundwater > 0.6 and state.season == "kharif":
            return "rice"
        if state.soil_type == "black":
            return "cotton"
        if state.groundwater < 0.35 or state.rainfall < 0.35:
            return "millet"
        if state.soil_type == "alluvial" and state.season == "rabi":
            return "wheat"
        if state.nitrogen < 0.3:
            return "pulses"
        return "maize"

    if decision_type == "irrigation":
        if state.groundwater < 0.4:
            return "drip"
        if state.chosen_crop in ["rice", "sugarcane"]:
            return "flood"
        if state.chosen_crop in ["millet", "pulses"] and state.rainfall > 0.6:
            return "rainfed"
        return "sprinkler"

    if decision_type == "fertilizer":
        if state.nitrogen < 0.3:
            return "nitrogen-rich"
        if state.phosphorus < 0.3:
            return "phosphorus-boost"
        if state.soil_health < 0.45:
            return "organic-compost"
        return "balanced-npk"

    if decision_type == "pest_control":
        if state.pest_risk > 0.75:
            return "chemical-pesticide"
        if state.pest_risk > 0.45:
            return "integrated-pest-management"
        return "biological-control"

    if decision_type == "strategy":
        if state.groundwater < 0.35:
            return "groundwater-conservation"
        if state.soil_health < 0.4:
            return "soil-restoration"
        return "maximize-yield"

    return "maize"


def choose_action_random(state, rng: random.Random) -> str:
    task = state.task
    step = state.step_index

    decision_sequence = {
        "crop-selection-easy": ["crop"],
        "farm-planning-medium": ["crop", "irrigation", "fertilizer"],
        "sustainable-farming-hard": ["crop", "irrigation", "fertilizer", "pest_control", "strategy"],
    }

    decision_type = decision_sequence[task][step]

    if decision_type == "crop":
        return rng.choice(CROPS)
    if decision_type == "irrigation":
        return rng.choice(IRRIGATION)
    if decision_type == "fertilizer":
        return rng.choice(FERTILIZERS)
    if decision_type == "pest_control":
        return rng.choice(PEST_CONTROL)
    if decision_type == "strategy":
        return rng.choice(STRATEGIES)

    return "maize"


# =========================================================
# Main OpenEnv Runner
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="OpenEnv Agriculture Inference Runner")
    parser.add_argument("--task", type=str, required=True, choices=[
        "crop-selection-easy",
        "farm-planning-medium",
        "sustainable-farming-hard",
    ])
    parser.add_argument("--policy", type=str, default="heuristic", choices=["heuristic", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="heuristic-baseline")
    parser.add_argument("--env", type=str, default="agriculture")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = AgricultureEnvironment(task_name=args.task, seed=args.seed)

    state = env.reset()

    print(f"[START] task={args.task} env={args.env} model={args.model}")

    done = False
    step_num = 0
    rewards = []
    final_score = 0.0
    success = False

    while not done:
        step_num += 1
        error_msg = "null"

        try:
            if args.policy == "heuristic":
                action_str = choose_action_heuristic(state)
            else:
                action_str = choose_action_random(state, rng)

            action = AgricultureAction(action=action_str)
            next_state, reward, done, info = env.step(action)

            rewards.append(reward)
            final_score = info.score
            success = info.success

            print(
                f"[STEP] step={step_num} "
                f"action={action_str} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error_msg}"
            )

            state = next_state

        except Exception as e:
            action_str = "invalid"
            reward = 0.0
            done = True
            error_msg = str(e).replace("\n", " ")

            print(
                f"[STEP] step={step_num} "
                f"action={action_str} "
                f"reward={reward:.2f} "
                f"done=true "
                f"error={error_msg}"
            )

            success = False
            final_score = 0.0
            rewards.append(reward)
            break

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_num} "
        f"score={final_score:.3f} "
        f"rewards={rewards_str}"
    )


if __name__ == "__main__":
    main()