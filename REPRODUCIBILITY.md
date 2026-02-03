# Reproducibility Guide for VEM Architecture

This document provides comprehensive instructions for reproducing and validating the Virtual Embodiment Module (VEM) architecture presented in "The Body is the Key: A Formal Architecture for Embodied Grounding in AGI."

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Architecture Validation](#architecture-validation)
4. [Minimal Demonstration](#minimal-demonstration)
5. [Proposed Experimental Validation](#proposed-experimental-validation)
6. [Code Structure](#code-structure)
7. [Known Limitations](#known-limitations)

---

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4 GB RAM
- CPU-only execution supported

### Recommended Requirements
- Python 3.10+
- 16 GB RAM
- NVIDIA GPU with CUDA support (for training)
- 50 GB disk space (for datasets and checkpoints)

### Operating Systems
- Linux (Ubuntu 20.04+, tested)
- macOS (10.15+)
- Windows 10/11 with WSL2

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/vem-implementation.git
cd vem-implementation
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv vem_env
source vem_env/bin/activate  # On Windows: vem_env\Scripts\activate

# Or using conda
conda create -n vem python=3.10
conda activate vem
```

### Step 3: Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

### Step 4: Verify Installation

```bash
# Run unit tests
python vem_core.py

# Run environment demo
python gridworld_env.py

# Run integration demo
python vem_gridworld_integration.py
```

Expected output: All scripts should complete without errors and display architecture information.

---

## Architecture Validation

### 1. Mathematical Consistency Check

Verify that the implementation matches the mathematical formalism in the paper.

```python
import torch
from vem_core import VirtualEmbodimentModule

# Initialize with known dimensions
vem = VirtualEmbodimentModule(
    d_visual=512, d_audio=128, d_proprio=64, d_homeo=32,
    d_attn=256, n_actions=10, memory_slots=100
)

# Create inputs
batch_size = 1
visual = torch.randn(batch_size, 64, 512)
audio = torch.randn(batch_size, 32, 128)
proprio = torch.randn(batch_size, 64)
homeo = torch.randn(batch_size, 32)

# Forward pass
output = vem(visual, audio, proprio, homeo, return_attention=True)

# Validate dimensions
assert output['action_probs'].shape == (batch_size, 10), "Action probs dimension mismatch"
assert output['exteroceptive_context'].shape == (batch_size, 256), "Extero context mismatch"
assert output['interoceptive_context'].shape == (batch_size, 256), "Intero context mismatch"

print("✓ All dimension checks passed")
```

### 2. Gradient Flow Verification

Ensure all components are differentiable and gradients flow correctly.

```python
# Enable gradient tracking
visual.requires_grad = True
audio.requires_grad = True
proprio.requires_grad = True
homeo.requires_grad = True

# Forward pass
output = vem(visual, audio, proprio, homeo)

# Compute dummy loss
loss = output['action_logits'].sum()

# Backward pass
loss.backward()

# Check gradients exist
assert visual.grad is not None, "Visual gradients not computed"
assert audio.grad is not None, "Audio gradients not computed"
assert proprio.grad is not None, "Proprio gradients not computed"
assert homeo.grad is not None, "Homeo gradients not computed"

print("✓ Gradient flow verified")
```

### 3. Attention Mechanism Validation

Verify attention weights sum to 1 and have correct shapes.

```python
output = vem(visual, audio, proprio, homeo, return_attention=True)

# Visual attention
visual_attn = output['attention_maps']['exteroceptive']['visual']
assert torch.allclose(visual_attn.sum(dim=-1), torch.ones_like(visual_attn.sum(dim=-1))), \
    "Visual attention doesn't sum to 1"

# Interoceptive attention
intero_attn = output['attention_maps']['interoceptive']
assert torch.allclose(intero_attn.sum(dim=-1), torch.ones_like(intero_attn.sum(dim=-1))), \
    "Interoceptive attention doesn't sum to 1"

print("✓ Attention mechanism validated")
```

### 4. Action Distribution Validation

Ensure action probabilities form valid probability distribution.

```python
action_probs = output['action_probs']

# Check sum to 1
assert torch.allclose(action_probs.sum(dim=-1), torch.ones(batch_size)), \
    "Action probabilities don't sum to 1"

# Check non-negative
assert (action_probs >= 0).all(), "Negative action probabilities detected"

# Check in valid range
assert (action_probs <= 1).all(), "Action probabilities exceed 1"

print("✓ Action distribution validated")
```

---

## Minimal Demonstration

### Running a Single Episode

```bash
python examples/run_episode.py --env gridworld --steps 100 --render
```

This demonstrates:
- Complete perception-action loop
- Multi-modal sensory processing
- Action selection via actuator attention
- Memory updates after each step

### Expected Behavior

The agent should:
1. Process visual and audio input through exteroceptive attention
2. Monitor internal state through interoceptive attention
3. Retrieve relevant memories
4. Select actions based on fused context
5. Update episodic memory with new experiences

Sample output:
```
Step 1: MOVE_FORWARD (confidence: 0.87)
  Energy: 99.9, Damage: 0.0
  Visual Attention: Focused on forward cell
  Action: Moved forward
  
Step 2: TURN_RIGHT (confidence: 0.45)
  Energy: 99.8, Damage: 0.0
  Visual Attention: Detecting hazard on right
  Action: Turned right
  
...
```

---

## Proposed Experimental Validation

Since full empirical validation requires extensive computational resources, we propose a **progressive validation roadmap**:

### Phase 1: Architecture Validation (Completed)
✓ Mathematical consistency  
✓ Gradient flow verification  
✓ Attention mechanism correctness  
✓ Action distribution validity  

### Phase 2: Minimal Environment Tests (Implementable without major resources)

#### Test 2.1: Object Permanence
**Hypothesis:** Agent should maintain object representation when occluded.

**Setup:**
```python
# Place food behind obstacle
env.place_object('food', x=5, y=5)
env.place_object('wall', x=4, y=5)  # Occludes food

# Agent observes food
agent.observe()

# Agent turns away
agent.turn(180)

# Agent turns back
agent.turn(180)

# Check memory: Does agent remember food location?
memory_content = agent.get_memory_attention()
assert memory_contains_food_location(memory_content), "Failed object permanence"
```

**Metric:** Percentage of trials where agent navigates to remembered location.

#### Test 2.2: Homeostatic Behavior
**Hypothesis:** Agent should seek food when energy is low.

**Setup:**
```python
# Deplete energy
agent.energy = 20.0

# Place food in view
env.place_object('food', x=agent.x+1, y=agent.y)

# Run for N steps
actions = []
for _ in range(10):
    action = agent.select_action()
    actions.append(action)
    agent.step(action)

# Verify food-seeking behavior
assert 'MOVE_FORWARD' in actions or 'INTERACT' in actions, \
    "Agent didn't seek food when hungry"
```

**Metric:** Frequency of food-seeking when energy < threshold.

#### Test 2.3: Hazard Avoidance
**Hypothesis:** Agent should avoid hazards after experiencing damage.

**Setup:**
```python
# Agent encounters hazard
agent.step_onto_hazard()  # Takes damage

# Place identical hazard nearby
env.place_object('hazard', nearby_position)

# Measure avoidance behavior
avoidance_rate = measure_hazard_avoidance(agent, n_trials=50)

assert avoidance_rate > random_baseline, "No learned hazard avoidance"
```

**Metric:** Avoidance rate compared to random policy.

### Phase 3: Comparative Baselines (Future Work)

Compare VEM against:

1. **Random Policy:** Sanity check
2. **Reactive Policy:** If energy < threshold → seek food
3. **Standard Transformer:** Same architecture without embodiment components
4. **Embodied Baselines:** Dreamer, World Models

**Proposed Metrics:**
- Average episode reward
- Survival time
- Sample efficiency (rewards per timestep)
- Transfer performance (train on GridWorld-A, test on GridWorld-B)

### Phase 4: Scaling to Complex Environments (Long-term)

1. **MuJoCo Humanoid**
   - 21 DoF humanoid body
   - Tasks: stand, walk, navigate
   
2. **PyBullet Manipulation**
   - Robot arm with gripper
   - Tasks: reach, grasp, place

3. **3D Virtual Environment**
   - First-person navigation
   - Multi-room exploration
   - Social interaction with NPCs

**Resource Requirements:**
- Compute: 4-8 GPUs for 1-2 weeks
- Storage: 500 GB for simulation data
- Personnel: 1-2 researchers

---

## Code Structure

```
vem-implementation/
│
├── vem_core.py                 # Core VEM architecture
├── gridworld_env.py            # Minimal test environment
├── vem_gridworld_integration.py # Integration example
│
├── requirements.txt            # Python dependencies
├── README.md                   # Quick start guide
├── REPRODUCIBILITY.md          # This file
│
├── examples/                   # Usage examples
│   ├── run_episode.py
│   ├── visualize_attention.py
│   └── train_agent.py
│
├── tests/                      # Unit tests
│   ├── test_attention.py
│   ├── test_memory.py
│   └── test_integration.py
│
└── docs/                       # Documentation
    ├── architecture.md
    ├── api_reference.md
    └── training_guide.md
```

---

## Known Limitations

### Current Implementation

1. **Simplified Environment:** GridWorld is 2D discrete; real embodiment requires continuous 3D physics

2. **Fixed Architecture:** Hyperparameters not optimized; defaults may be suboptimal

3. **Memory Mechanism:** Current implementation uses simple write strategy; should use attention-based addressing (NTM/DNC style)

4. **No Transfer Learning:** No mechanism to transfer knowledge across environments

5. **Single Agent:** No multi-agent interaction or social learning

### Addressing Limitations

Future improvements should:

1. **Integrate Physics Engines:**
   ```python
   import mujoco
   from dm_control import suite
   
   env = suite.load(domain_name="humanoid", task_name="stand")
   ```

2. **Implement Advanced Memory:**
   ```python
   # Differentiable Neural Computer style addressing
   class NTMMemory(nn.Module):
       def write(self, key, value, erase_vector, add_vector):
           # Content-based + location-based addressing
           # ...
   ```

3. **Add Curriculum Learning:**
   ```python
   curriculum = [
       ('simple_nav', 1000),
       ('food_collection', 2000),
       ('hazard_avoidance', 3000),
       ('goal_reaching', 5000)
   ]
   ```

4. **Enable Multi-Modal Scaling:**
   - Integrate vision transformers for visual input
   - Add tactile sensors (force, texture)
   - Implement olfactory perception

---

## Data Availability Statement

**Recommended for Paper:**

> "This work presents a theoretical framework with a reference implementation. Code implementing the Virtual Embodiment Module architecture is available at [GitHub URL]. The implementation includes:
> 
> 1. Complete PyTorch implementation of VEM architecture (`vem_core.py`)
> 2. Minimal 2D GridWorld environment for validation (`gridworld_env.py`)
> 3. Integration examples demonstrating perception-action loop
> 4. Unit tests verifying mathematical correctness
> 5. Reproducibility guide with validation protocols
> 
> No pre-trained models or experimental datasets are provided as this work focuses on architectural design. The codebase enables researchers to:
> - Instantiate the VEM architecture with custom configurations
> - Validate mathematical formalism through gradient checks
> - Integrate with existing simulation environments (MuJoCo, PyBullet, Isaac Gym)
> - Extend to novel sensory modalities and action spaces
> 
> Requirements: Python 3.8+, PyTorch 2.0+, NumPy. Full installation and usage instructions in repository README."

---

## Citation

If you use this implementation, please cite:

```bibtex
@article{quiroga2025body,
  title={The Body is the Key: A Formal Architecture for Embodied Grounding in Artificial General Intelligence},
  author={Quiroga Rodríguez, Elio},
  journal={Array},
  year={2025},
  note={Under Review}
}

@software{quiroga2025vem,
  author = {Quiroga Rodríguez, Elio},
  title = {Virtual Embodiment Module: Reference Implementation},
  year = {2025},
  url = {https://github.com/your-username/vem-implementation},
  version = {1.0.0}
}
```

---

## Contact & Support

For questions about reproducibility:
- **Email:** elio.quiroga@pdi.atlanticomedio.es
- **Issues:** GitHub issue tracker
- **Discussions:** GitHub discussions

---

## Acknowledgments

This implementation builds upon:
- Vaswani et al. (2017) - Transformer architecture
- Pfeifer & Bongard (2006) - Embodied AI principles
- Hafner et al. (2020) - World Models (Dreamer)
- Graves et al. (2014) - Neural Turing Machines

We thank the open-source community for PyTorch and related tools.

---

**Last Updated:** February 2025  
**Version:** 1.0.0
