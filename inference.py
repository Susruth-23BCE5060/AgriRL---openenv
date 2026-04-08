"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    API_KEY        Your API key (or HF_TOKEN as fallback).
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import random
import textwrap
from typing import List, Optional

from openai import OpenAI

from server.agriculture_environment import AgricultureEnvironment
from models import AgricultureAction

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
TASK_NAME = os.getenv("TASK", "crop-selection-easy")
BENCHMARK = os.getenv("ENV", "agriculture")
MAX_STEPS = 5  # Max steps for hardest task
TEMPERATURE = 0.1
MAX_TOKENS = 50
SUCCESS_SCORE_THRESHOLD = 0.7  # Based on baseline results

# Action spaces
CROPS = ["rice", "wheat", "maize", "cotton", "millet", "sugarcane", "pulses"]
IRRIGATION = ["drip", "sprinkler", "flood", "rainfed"]
FERTILIZERS = ["nitrogen-rich", "balanced-npk", "organic-compost", "phosphorus-boost"]
PEST_CONTROL = ["none", "integrated-pest-management", "chemical-pesticide", "biological-control"]
STRATEGIES = ["groundwater-conservation", "maximize-yield", "low-cost-farming", "soil-restoration"]

# Initialize OpenAI client
if API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
else:
    client = None

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert agricultural advisor. Based on the farm conditions provided, recommend the best action for the current decision type.
    Reply with exactly the action name — no quotes, no prefixes, just the action text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(state, decision_type: str) -> str:
    return textwrap.dedent(
        f"""
        Farm State:
        - Task: {state.task}
        - Step: {state.step_index + 1}
        - Soil Type: {state.soil_type}
        - Nitrogen: {state.nitrogen}
        - Phosphorus: {state.phosphorus}
        - Potassium: {state.potassium}
        - Rainfall: {state.rainfall}
        - Temperature: {state.temperature}
        - Groundwater: {state.groundwater}
        - Pest Risk: {state.pest_risk}
        - Soil Health: {state.soil_health}
        - Season: {state.season}
        - Chosen Crop: {state.chosen_crop or 'None'}
        - Chosen Irrigation: {state.chosen_irrigation or 'None'}
        - Chosen Fertilizer: {state.chosen_fertilizer or 'None'}
        - Chosen Pest Control: {state.chosen_pest_control or 'None'}
        - Chosen Strategy: {state.chosen_strategy or 'None'}

        Current Decision: {decision_type}

        Available options: {get_options_for_decision(decision_type)}

        Recommend the best {decision_type} for this farm.
        """
    ).strip()


def get_options_for_decision(decision_type: str) -> List[str]:
    if decision_type == "crop":
        return CROPS
    elif decision_type == "irrigation":
        return IRRIGATION
    elif decision_type == "fertilizer":
        return FERTILIZERS
    elif decision_type == "pest_control":
        return PEST_CONTROL
    elif decision_type == "strategy":
        return STRATEGIES
    return []


def get_model_message(state, decision_type: str) -> str:
    if client is None:
        # Fallback to random if no client
        return random.choice(get_options_for_decision(decision_type))

    user_prompt = build_user_prompt(state, decision_type)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        options = get_options_for_decision(decision_type)
        if text in options:
            return text
        else:
            return random.choice(options)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return random.choice(get_options_for_decision(decision_type))


async def main() -> None:
    env = AgricultureEnvironment(task_name=TASK_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset_async()
        state = result

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Determine decision type
            decision_sequence = {
                "crop-selection-easy": ["crop"],
                "farm-planning-medium": ["crop", "irrigation", "fertilizer"],
                "sustainable-farming-hard": ["crop", "irrigation", "fertilizer", "pest_control", "strategy"],
            }
            current_decision = decision_sequence[TASK_NAME][state.step_index]

            action_str = get_model_message(state, current_decision)

            result = await env.step_async(AgricultureAction(action=action_str))
            obs = result

            reward = obs.reward or 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            state = obs

            if done:
                break

        # Calculate final score (normalized to [0,1])
        if obs.info:
            score = obs.info.get('score', 0.0)
        else:
            score = 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())