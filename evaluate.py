"""
Evaluation script for Agriculture OpenEnv benchmark.

This is for developer-side benchmarking and comparison.
It is NOT constrained to the strict OpenEnv stdout format.
"""

import argparse
from collections import Counter, defaultdict

from server.agriculture_environment import AgricultureEnvironment, TASKS
from models import AgricultureAction


# =========================================================
# Policies
# =========================================================
def choose_action(env: AgricultureEnvironment, policy: str) -> str:
    s = env.state()
    step_idx = s.step_index
    task = env.task_config
    decision_type = task.decision_sequence[step_idx]

    if policy == "random":
        return random_policy(decision_type, env)

    if policy == "heuristic":
        return heuristic_policy(env, decision_type)

    raise ValueError(f"Unknown policy: {policy}")


def random_policy(decision_type: str, env: AgricultureEnvironment) -> str:
    rng = env.rng

    if decision_type == "crop":
        return rng.choice(["rice", "wheat", "maize", "cotton", "millet", "sugarcane", "pulses"])
    elif decision_type == "irrigation":
        return rng.choice(["drip", "sprinkler", "flood", "rainfed"])
    elif decision_type == "fertilizer":
        return rng.choice(["nitrogen-rich", "balanced-npk", "organic-compost", "phosphorus-boost"])
    elif decision_type == "pest_control":
        return rng.choice(["none", "integrated-pest-management", "chemical-pesticide", "biological-control"])
    elif decision_type == "strategy":
        return rng.choice(["groundwater-conservation", "maximize-yield", "low-cost-farming", "soil-restoration"])

    return "maize"


def heuristic_policy(env: AgricultureEnvironment, decision_type: str) -> str:
    s = env.state()

    if decision_type == "crop":
        candidates = ["rice", "wheat", "maize", "cotton", "millet", "sugarcane", "pulses"]

        best_crop = None
        best_score = -1

        for crop in candidates:
            soil_score = {
                "loamy": {"wheat": 0.9, "maize": 0.85, "pulses": 0.8, "cotton": 0.7, "rice": 0.65, "millet": 0.75, "sugarcane": 0.7},
                "clay": {"rice": 0.95, "sugarcane": 0.8, "cotton": 0.75, "wheat": 0.7, "maize": 0.65, "millet": 0.55, "pulses": 0.6},
                "sandy": {"millet": 0.9, "pulses": 0.8, "cotton": 0.75, "maize": 0.65, "wheat": 0.55, "rice": 0.35, "sugarcane": 0.4},
                "black": {"cotton": 0.95, "millet": 0.8, "sugarcane": 0.75, "pulses": 0.7, "maize": 0.7, "wheat": 0.65, "rice": 0.55},
                "alluvial": {"rice": 0.9, "wheat": 0.9, "sugarcane": 0.85, "maize": 0.8, "pulses": 0.75, "cotton": 0.7, "millet": 0.65},
            }[s.soil_type][crop]

            season_score = {
                "kharif": {"rice": 0.95, "maize": 0.85, "cotton": 0.85, "millet": 0.8, "sugarcane": 0.75, "pulses": 0.7, "wheat": 0.4},
                "rabi": {"wheat": 0.95, "pulses": 0.85, "maize": 0.7, "millet": 0.7, "sugarcane": 0.65, "cotton": 0.45, "rice": 0.5},
                "zaid": {"maize": 0.8, "millet": 0.8, "pulses": 0.75, "sugarcane": 0.7, "rice": 0.6, "cotton": 0.6, "wheat": 0.35},
            }[s.season][crop]

            water_need = {
                "rice": 0.95, "sugarcane": 0.9, "cotton": 0.75,
                "maize": 0.65, "wheat": 0.6, "pulses": 0.4, "millet": 0.25,
            }[crop]

            climate_pref = {
                "rice": 0.7, "wheat": 0.4, "maize": 0.65,
                "cotton": 0.85, "millet": 0.8, "sugarcane": 0.75, "pulses": 0.55,
            }[crop]

            nitrogen_need = {
                "rice": 0.75, "wheat": 0.7, "maize": 0.8,
                "cotton": 0.65, "millet": 0.35, "sugarcane": 0.85, "pulses": 0.25,
            }[crop]

            score = (
                0.30 * soil_score +
                0.25 * season_score +
                0.20 * (1 - abs(water_need - s.groundwater)) +
                0.15 * (1 - abs(climate_pref - s.temperature)) +
                0.10 * (1 - abs(nitrogen_need - s.nitrogen))
            )

            if score > best_score:
                best_score = score
                best_crop = crop

        return best_crop

    elif decision_type == "irrigation":
        if s.groundwater < 0.35:
            return "drip"
        if s.chosen_crop in ["rice", "sugarcane"]:
            return "sprinkler" if s.groundwater < 0.6 else "flood"
        if s.chosen_crop in ["millet", "pulses"]:
            return "rainfed" if s.rainfall > 0.55 else "drip"
        return "drip"

    elif decision_type == "fertilizer":
        n_def = 1 - s.nitrogen
        p_def = 1 - s.phosphorus
        k_def = 1 - s.potassium

        if s.soil_health < 0.45:
            return "organic-compost"
        if n_def > p_def and n_def > k_def:
            return "nitrogen-rich"
        if p_def > n_def and p_def > k_def:
            return "phosphorus-boost"
        return "balanced-npk"

    elif decision_type == "pest_control":
        if s.pest_risk > 0.75:
            return "integrated-pest-management"
        if s.pest_risk > 0.55:
            return "biological-control"
        return "none"

    elif decision_type == "strategy":
        if s.groundwater < 0.35:
            return "groundwater-conservation"
        if s.soil_health < 0.4:
            return "soil-restoration"
        return "low-cost-farming"

    return "maize"


# =========================================================
# Evaluation
# =========================================================
def run_episode(task_name: str, policy: str, seed: int):
    env = AgricultureEnvironment(task_name=task_name, seed=seed)
    env.reset()

    rewards = []
    actions = []
    done = False
    final_score = 0.0
    success = False

    while not done:
        action_str = choose_action(env, policy)
        actions.append(action_str)

        _, reward, done, info = env.step(AgricultureAction(action=action_str))
        rewards.append(reward)
        final_score = info.score
        success = info.success

    return {
        "rewards": rewards,
        "actions": actions,
        "score": final_score,
        "success": success,
        "total_reward": sum(rewards),
    }


def evaluate_task(task_name: str, policy: str, episodes: int, base_seed: int):
    total_rewards = []
    total_scores = []
    successes = 0
    action_counter = Counter()
    sample_episode = None

    for i in range(episodes):
        result = run_episode(task_name, policy, base_seed + i)

        total_rewards.append(result["total_reward"])
        total_scores.append(result["score"])
        successes += int(result["success"])
        action_counter.update(result["actions"])

        if sample_episode is None:
            sample_episode = result

    avg_reward = sum(total_rewards) / episodes
    avg_score = sum(total_scores) / episodes
    success_rate = successes / episodes

    print("\n" + "=" * 70)
    print(f"TASK: {task_name}")
    print(f"POLICY: {policy}")
    print(f"EPISODES: {episodes}")
    print("-" * 70)
    print(f"Average Reward : {avg_reward:.4f}")
    print(f"Average Score  : {avg_score:.4f}")
    print(f"Success Rate   : {success_rate:.2%}")
    print("-" * 70)
    print("Most Common Actions:")
    for action, count in action_counter.most_common(10):
        print(f"  {action:<30} {count}")
    print("-" * 70)
    print("Sample Trajectory:")
    for i, (a, r) in enumerate(zip(sample_episode["actions"], sample_episode["rewards"]), start=1):
        done = (i == len(sample_episode["actions"]))
        score = sample_episode["score"] if done else r
        print(f"  Step {i}: action={a}, reward={r:.4f}, score={score:.4f}, done={done}")
    print("=" * 70)

    return {
        "avg_reward": avg_reward,
        "avg_score": avg_score,
        "success_rate": success_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all", help="Task name or 'all'")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--policy", type=str, default="heuristic", choices=["heuristic", "random"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.task == "all":
        task_names = list(TASKS.keys())
    else:
        if args.task not in TASKS:
            raise ValueError(f"Unknown task: {args.task}")
        task_names = [args.task]

    print(f"[START] policy={args.policy} episodes={args.episodes} seed={args.seed}")

    summary = {}
    for task_name in task_names:
        summary[task_name] = evaluate_task(task_name, args.policy, args.episodes, args.seed)

    print("\n[END] Evaluation complete.")
    print("\n" + "#" * 70)
    print("OVERALL COMPARISON")
    print("#" * 70)
    for task_name, stats in summary.items():
        print(
            f"{task_name:<30} | "
            f"avg_score={stats['avg_score']:.4f} | "
            f"success_rate={stats['success_rate']:.2%}"
        )


if __name__ == "__main__":
    main()