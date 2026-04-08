---
title: Agriculture Environment Server
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - agriculture
  - benchmark
---

# Agriculture Environment

A multi-step **OpenEnv benchmark environment** for agricultural decision-making.

This environment simulates realistic farm planning tasks where an agent must choose crops and farming strategies based on:

- **Soil type**
- **Nutrient levels**
- **Rainfall**
- **Temperature**
- **Groundwater availability**
- **Pest risk**
- **Soil health**
- **Season**

The benchmark is designed for **reinforcement learning, planning agents, and decision-making systems**, with tasks ranging from simple crop recommendation to sustainable multi-step farm planning.

---

# Why this environment exists

Agricultural planning is a real-world sequential decision problem.

Farmers and agricultural planners do not make a single isolated choice — they make a sequence of interconnected decisions such as:

- Which crop to grow
- Which irrigation method to use
- What fertilizer plan to follow
- How to manage pest risk
- Whether to optimize for yield, cost, or sustainability

This environment turns that real-world process into a structured benchmark for **LLM agents, RL agents, and planning systems**.

---

# Benchmark Tasks

The environment includes **3 difficulty levels** with increasing decision complexity.

## 1) `crop-selection-easy`
**Goal:** Select the most suitable crop for a given farm state.

**Decisions:**
- Crop only

**Episode Length:** 1 step

---

## 2) `farm-planning-medium`
**Goal:** Build a farm plan by selecting a crop, irrigation strategy, and fertilizer.

**Decisions:**
- Crop
- Irrigation
- Fertilizer

**Episode Length:** 3 steps

---

## 3) `sustainable-farming-hard`
**Goal:** Build a full sustainable farm plan balancing productivity and long-term land health.

**Decisions:**
- Crop
- Irrigation
- Fertilizer
- Pest control
- Long-term farming strategy

**Episode Length:** 5 steps

---

# State Space

Each episode starts with a generated farm state containing environmental and agronomic signals:

- `soil_type`
- `nitrogen`
- `phosphorus`
- `potassium`
- `rainfall`
- `temperature`
- `groundwater`
- `pest_risk`
- `soil_health`
- `season`

These values influence which actions are most suitable.

---

# Action Space

Depending on the current task step, the agent selects one action from a domain-specific set.

## Crop options
- `rice`
- `wheat`
- `maize`
- `cotton`
- `millet`
- `sugarcane`
- `pulses`

## Irrigation options
- `drip`
- `sprinkler`
- `flood`
- `rainfed`

## Fertilizer options
- `nitrogen-rich`
- `balanced-npk`
- `organic-compost`
- `phosphorus-boost`

## Pest control options
- `none`
- `integrated-pest-management`
- `chemical-pesticide`
- `biological-control`

## Long-term strategy options
- `groundwater-conservation`
- `maximize-yield`
- `low-cost-farming`
- `soil-restoration`

---

# Reward Design

The environment uses a **meaningful shaped reward function** with **partial progress signals** at every step.

Rewards are based on domain-inspired compatibility such as:

- soil-crop suitability
- season-crop suitability
- water requirement fit
- climate suitability
- nutrient correction quality
- irrigation efficiency
- pest protection effectiveness
- ecological safety
- sustainability and long-term soil/groundwater outcomes

This allows agents to receive feedback **before the final step**, making the environment suitable for:

- reinforcement learning
- heuristic planning
- LLM action selection
- reward-based benchmarking

---

# Success Criteria

Each task has a final score and success threshold:

- **Easy:** success if final score ≥ `0.70`
- **Medium:** success if final score ≥ `0.72`
- **Hard:** success if final score ≥ `0.70`

This ensures tasks are not purely random and reward quality matters.

---

# OpenEnv Compliance

This project implements the required **OpenEnv benchmark spec**, including:

- typed environment models
- `reset()`
- `step()`
- `state()`
- `openenv.yaml`
- reproducible inference script
- multi-task benchmark setup
- shaped rewards with partial progress

---

# Quick Start

## 1) Clone the project

```bash
git clone <your-repo-url>
cd agriculture
````

---

## 2) Install dependencies

If you are using `uv`:

```bash
uv sync
```

Or with pip:

```bash
pip install -r server/requirements.txt
```

---

## 3) Run a baseline episode

### Heuristic baseline

```bash
python inference.py --task crop-selection-easy --policy heuristic
```

### Random baseline

```bash
python inference.py --task farm-planning-medium --policy random --model random-baseline
```

### LLM baseline

```bash
python inference.py --task sustainable-farming-hard --policy llm --model gpt-4o-mini
```

---

# Inference Script (Benchmark Mode)

The `inference.py` script follows the required benchmark-style stdout format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

## Example

```bash
python inference.py --task sustainable-farming-hard --policy heuristic
```

Example output:

```text
[START] task=sustainable-farming-hard env=agriculture model=heuristic-baseline
[STEP] step=1 action=maize reward=0.83 done=false error=null
[STEP] step=2 action=drip reward=0.73 done=false error=null
[STEP] step=3 action=organic-compost reward=0.97 done=false error=null
[STEP] step=4 action=none reward=0.81 done=false error=null
[STEP] step=5 action=low-cost-farming reward=0.67 done=true error=null
[END] success=true steps=5 score=0.708 rewards=0.83,0.73,0.97,0.81,0.67
```

---

# Evaluation Script (Developer / Research Mode)

The `evaluate.py` script is used for comparing policies over many episodes.

## Run heuristic baseline

```bash
python evaluate.py --task all --episodes 100 --policy heuristic
```

## Run random baseline

```bash
python evaluate.py --task all --episodes 100 --policy random
```

This prints:

* average reward
* average score
* success rate
* most common actions
* sample trajectories

---

# Baseline Results

Results below were generated with **100 episodes** and **seed=42**.

## Heuristic Baseline

| Task                       | Avg Score  | Success Rate |
| -------------------------- | ---------- | ------------ |
| `crop-selection-easy`      | **0.8747** | **100.00%**  |
| `farm-planning-medium`     | **0.7798** | **86.00%**   |
| `sustainable-farming-hard` | **0.7120** | **62.00%**   |

## Random Baseline

| Task                       | Avg Score  | Success Rate |
| -------------------------- | ---------- | ------------ |
| `crop-selection-easy`      | **0.7665** | **72.00%**   |
| `farm-planning-medium`     | **0.6958** | **28.00%**   |
| `sustainable-farming-hard` | **0.6322** | **17.00%**   |

---

# Interpretation of Results

These benchmark results show that:

* **Task difficulty increases meaningfully** from easy → medium → hard
* **Random policies perform worse**
* **Heuristic policies perform better**
* The environment is **not trivial**
* The reward function provides **useful learning signal**

This makes the benchmark suitable for testing:

* hand-designed policies
* LLM-based decision agents
* reinforcement learning algorithms
* planning systems

---

# Using the Environment Programmatically

You can also interact with the environment directly in Python.

## Example

```python
from server.agriculture_environment import AgricultureEnvironment
from models import AgricultureAction

env = AgricultureEnvironment(task_name="farm-planning-medium", seed=42)

state = env.reset()
print("Initial state:", state)

done = False
while not done:
    # Example manual policy
    action = AgricultureAction(action="maize")
    state, reward, done, info = env.step(action)

    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
```

---

# Running the API Server

To run the environment locally as an OpenEnv-compatible server:

```bash
uvicorn server.app:app --reload
```

Once running, the server exposes:

* **Web Interface** at `/web`
* **API Docs** at `/docs`
* **Health Check** at `/health`
* **WebSocket Endpoint** at `/ws`

---

# Docker

## Build Docker image

```bash
docker build -t agriculture-env:latest -f Dockerfile .
```

## Run container

```bash
docker run -p 8000:8000 agriculture-env:latest
```

---

# Deploying to Hugging Face Spaces

You can deploy this environment directly as an OpenEnv environment.

## Push to Hugging Face

```bash
openenv push
```

Or:

```bash
openenv push --repo-id my-org/agriculture-env --private
```

After deployment, your environment will be available as a Docker Space with:

* interactive web UI
* API documentation
* health endpoints
* WebSocket session support

---

# Project Structure

```text
│   .env
│   .env.example
│   .gitignore
│   client.py
│   Dockerfile
│   evaluate.py
│   inference.py
│   models.py
│   openenv.yaml
│   pyproject.toml
│   README.md
│   uv.lock
│   __init__.py
│   
├───server
│   │   agriculture_environment.py
│   │   app.py
│   │   requirements.txt
│   │   __init__.py
│   │   
│   └───__pycache__
│           agriculture_environment.cpython-310.pyc
│           app.cpython-310.pyc
│           __init__.cpython-310.pyc
│           
└───__pycache__
        client.cpython-310.pyc
        models.cpython-310.pyc
        __init__.cpython-310.pyc
```

---