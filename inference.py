"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    API_KEY        Your API key (or HF_TOKEN as fallback).
    IMAGE_NAME     The name of the local image to use for the environment if using from_docker_image()

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import argparse
import os
import random
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN"))  # Fallback to HF_TOKEN if API_KEY not set

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Initialize OpenAI client with provided base_url and api_key
if API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
else:
    client = None

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
# LLM-Based Policy
# =========================================================
def choose_action_llm(state) -> str:
    task = state.task
    step = state.step_index

    decision_sequence = {
        "crop-selection-easy": ["crop"],
        "farm-planning-medium": ["crop", "irrigation", "fertilizer"],
        "sustainable-farming-hard": ["crop", "irrigation", "fertilizer", "pest_control", "strategy"],
    }

    decision_type = decision_sequence[task][step]

    # Build prompt based on decision type
    if decision_type == "crop":
        options = CROPS
        prompt = f"""
You are an agricultural expert. Based on the following farm conditions, recommend the best crop to plant:

Farm State:
- Soil Type: {state.soil_type}
- Nitrogen Level: {state.nitrogen}
- Phosphorus Level: {state.phosphorus}
- Potassium Level: {state.potassium}
- Rainfall: {state.rainfall}
- Temperature: {state.temperature}
- Groundwater: {state.groundwater}
- Pest Risk: {state.pest_risk}
- Soil Health: {state.soil_health}
- Season: {state.season}

Available crops: {', '.join(CROPS)}

Respond with only the crop name, nothing else.
"""
    elif decision_type == "irrigation":
        options = IRRIGATION
        prompt = f"""
You are an agricultural expert. The farmer has chosen {state.chosen_crop} as the crop. Based on the farm conditions, recommend the best irrigation method:

Farm State:
- Chosen Crop: {state.chosen_crop}
- Soil Type: {state.soil_type}
- Rainfall: {state.rainfall}
- Groundwater: {state.groundwater}
- Pest Risk: {state.pest_risk}
- Soil Health: {state.soil_health}

Available irrigation methods: {', '.join(IRRIGATION)}

Respond with only the irrigation method name, nothing else.
"""
    elif decision_type == "fertilizer":
        options = FERTILIZERS
        prompt = f"""
You are an agricultural expert. The farmer has chosen {state.chosen_crop} as the crop and {state.chosen_irrigation} as irrigation. Recommend the best fertilizer:

Farm State:
- Chosen Crop: {state.chosen_crop}
- Chosen Irrigation: {state.chosen_irrigation}
- Soil Type: {state.soil_type}
- Nitrogen Level: {state.nitrogen}
- Phosphorus Level: {state.phosphorus}
- Potassium Level: {state.potassium}
- Soil Health: {state.soil_health}

Available fertilizers: {', '.join(FERTILIZERS)}

Respond with only the fertilizer name, nothing else.
"""
    elif decision_type == "pest_control":
        options = PEST_CONTROL
        prompt = f"""
You are an agricultural expert. Recommend the best pest control method:

Farm State:
- Pest Risk: {state.pest_risk}
- Soil Health: {state.soil_health}

Available pest control methods: {', '.join(PEST_CONTROL)}

Respond with only the pest control method name, nothing else.
"""
    elif decision_type == "strategy":
        options = STRATEGIES
        prompt = f"""
You are an agricultural expert. Recommend the best long-term farming strategy:

Farm State:
- Groundwater: {state.groundwater}
- Soil Health: {state.soil_health}

Available strategies: {', '.join(STRATEGIES)}

Respond with only the strategy name, nothing else.
"""
    else:
        return "maize"

    try:
        if client is None:
            raise Exception("OpenAI client not initialized")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert agricultural advisor."},
                {"role": "user", "content": prompt.strip()},
            ],
            max_tokens=50,
            temperature=0.1,
        )
        action = response.choices[0].message.content.strip()
        # Validate that the action is in options
        if action in options:
            return action
        else:
            # Fallback to random if invalid
            return random.choice(options)
    except Exception as e:
        # Fallback to random on error
        return random.choice(options)


# =========================================================
# Main OpenEnv Runner
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="OpenEnv Agriculture Inference Runner")
    parser.add_argument(
        "--task",
        type=str,
        default=os.getenv("TASK", "crop-selection-easy"),
        choices=[
            "crop-selection-easy",
            "farm-planning-medium",
            "sustainable-farming-hard",
        ],
        help="Task to run (can also be set via TASK env variable)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=os.getenv("POLICY", "heuristic"),
        choices=["heuristic", "random", "llm"],
        help="Policy to run (can also be set via POLICY env variable)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", "42")),
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL", "heuristic-baseline"),
        help="Model name to report in logs (can also be set via MODEL env variable)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=os.getenv("ENV", "agriculture"),
        help="Environment name to report in logs (can also be set via ENV env variable)",
    )

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
            elif args.policy == "random":
                action_str = choose_action_random(state, rng)
            elif args.policy == "llm":
                action_str = choose_action_llm(state)
            else:
                action_str = choose_action_heuristic(state)

            action = AgricultureAction(action=action_str)
            obs = env.step(action)
            reward = obs.reward or 0.0
            done = obs.done
            info = obs.info or {}

            rewards.append(reward)
            final_score = info.get('score', 0.0)
            success = info.get('success', False)

            print(
                f"[STEP] step={step_num} "
                f"action={action_str} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error_msg}"
            )

            state = obs

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