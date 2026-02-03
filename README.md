# Virtual Embodiment Module (VEM) - Reference Implementation

This repository contains a reference implementation of the **Virtual Embodiment Module (VEM)** as described in the paper:

> **"The Body is the Key: A Formal Architecture for Embodied Grounding in Artificial General Intelligence"**  
> Elio Quiroga Rodríguez, Universidad del Atlántico Medio

## Overview

The VEM is a novel neural architecture that extends the transformer paradigm to embodied cognition. Unlike traditional transformers that operate on symbolic sequences, the VEM processes continuous, multi-modal sensory streams from a simulated body, integrating:

1. **Exteroceptive Attention** - Processing external sensory inputs (vision, audition)
2. **Interoceptive Attention** - Monitoring internal body states (proprioception, homeostasis)
3. **Actuator Attention** - Selecting actions based on attended context
4. **Episodic Memory** - Maintaining temporal continuity and world state

## Architecture Components

### Core Modules

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| `ExteroceptiveAttention` | Process external senses | Visual features, Audio features | Exteroceptive context vector |
| `InteroceptiveAttention` | Monitor internal state | Proprioception, Homeostasis, History | Interoceptive context vector |
| `ActuatorAttention` | Select actions | Situation context | Action probability distribution |
| `EpisodicMemory` | Store/retrieve experiences | Context query | Retrieved memory context |
| `VirtualEmbodimentModule` | Complete integration | All sensory inputs | Action policy + contexts |

### Mathematical Foundation

The VEM implements the formalism from Section "The Formalism" of the paper:

**Sensory State:**
```
S_t = [V_t; A_t; P_t; H_t]
```
where:
- V_t: Visual sensory vector (∈ ℝ^d_v)
- A_t: Auditory sensory vector (∈ ℝ^d_a)
- P_t: Proprioceptive vector (∈ ℝ^d_p)
- H_t: Homeostatic vector (∈ ℝ^d_h)

**Attention Mechanisms:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Action Policy:**
```
π(a_t | S_t) = softmax((Q_act · K_act^T) / τ)
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/vem-implementation.git
cd vem-implementation

# Install dependencies
pip install torch numpy matplotlib
```

## Quick Start

```python
import torch
from vem_core import VirtualEmbodimentModule

# Initialize VEM
vem = VirtualEmbodimentModule(
    d_visual=512,      # Visual feature dimension
    d_audio=128,       # Audio feature dimension
    d_proprio=64,      # Proprioceptive state dimension
    d_homeo=32,        # Homeostatic state dimension
    d_attn=256,        # Attention dimension
    n_actions=10,      # Number of action primitives
    memory_slots=100   # Episodic memory capacity
)

# Create sensory input (batch_size=4)
visual = torch.randn(4, 64, 512)        # 64 visual regions
audio = torch.randn(4, 32, 128)         # 32 audio bins
proprioception = torch.randn(4, 64)     # Joint angles, etc.
homeostasis = torch.randn(4, 32)        # Energy, pain, etc.

# Forward pass
output = vem(visual, audio, proprioception, homeostasis)

# Get action probabilities
action_probs = output['action_probs']   # [batch, n_actions]

# Sample action
action = torch.distributions.Categorical(action_probs).sample()
```

## Usage Examples

### 1. Basic Perception-Action Loop

```python
# Initialize environment and agent
env = SimulatedEnvironment()
vem = VirtualEmbodimentModule(...)

for timestep in range(1000):
    # Get sensory input from environment
    visual, audio, proprio, homeo = env.get_observations()
    
    # Process through VEM
    output = vem(visual, audio, proprio, homeo)
    
    # Sample action from policy
    action = torch.distributions.Categorical(
        output['action_probs']
    ).sample()
    
    # Execute in environment
    env.step(action)
    
    # Update episodic memory
    action_onehot = F.one_hot(action, vem.n_actions).float()
    vem.update_memory(output['situation_context'], action_onehot)
```

### 2. Attention Visualization

```python
# Get attention maps
output = vem(visual, audio, proprio, homeo, return_attention=True)

# Extract attention weights
visual_attn = output['attention_maps']['exteroceptive']['visual']
audio_attn = output['attention_maps']['exteroceptive']['audio']
intero_attn = output['attention_maps']['interoceptive']
memory_attn = output['attention_maps']['memory']

# Visualize where the agent is "looking"
import matplotlib.pyplot as plt
plt.imshow(visual_attn[0, 0].reshape(8, 8).detach())
plt.title('Visual Attention Heatmap')
plt.colorbar()
plt.show()
```

### 3. Internal State Monitoring

```python
# Track homeostatic signals over time
homeostatic_history = []

for t in range(100):
    visual, audio, proprio, homeo = env.get_observations()
    output = vem(visual, audio, proprio, homeo)
    
    # Log internal attention
    intero_context = output['interoceptive_context']
    homeostatic_history.append(intero_context.detach())
    
    # Check if homeostasis violated (e.g., pain signal)
    if homeo[0, -1] > threshold:  # Last dim = pain
        print(f"Timestep {t}: Pain signal detected!")
```

## Model Specifications

### Default Configuration

```python
{
    'd_visual': 512,        # Visual feature dimension
    'd_audio': 128,         # Audio feature dimension
    'd_proprio': 64,        # Proprioceptive dimension
    'd_homeo': 32,          # Homeostatic dimension
    'd_attn': 256,          # Attention hidden dimension
    'n_actions': 10,        # Action repertoire size
    'd_action': 128,        # Action embedding dimension
    'memory_slots': 100,    # Episodic memory capacity
}
```

### Parameter Count

With default configuration:
- **Total Parameters:** ~2.5M
- **Trainable Parameters:** ~2.5M

### Computational Complexity

- **Exteroceptive Attention:** O(N²) where N = number of visual regions
- **Interoceptive Attention:** O(W²) where W = memory window size
- **Actuator Attention:** O(M) where M = number of actions
- **Memory Retrieval:** O(L) where L = memory slots

## Input Specifications

### Visual Input
- **Shape:** `[batch, n_visual_regions, d_visual]`
- **Example:** 64 regions (8×8 grid) × 512 dimensions
- **Source:** CNN feature extractor applied to egocentric view

### Audio Input
- **Shape:** `[batch, n_audio_bins, d_audio]`
- **Example:** 32 time-frequency bins × 128 dimensions
- **Source:** Spectrogram or learned audio features

### Proprioceptive Input
- **Shape:** `[batch, d_proprio]`
- **Example:** 64 dimensions
- **Content:** Joint angles, muscle tensions, end-effector positions

### Homeostatic Input
- **Shape:** `[batch, d_homeo]`
- **Example:** 32 dimensions
- **Content:** Energy level, damage signals, temperature, comfort

## Output Specifications

### Action Distribution
- **Shape:** `[batch, n_actions]`
- **Type:** Probability distribution (sums to 1.0)
- **Usage:** Sample actions or use for policy gradient

### Context Vectors
- `situation_context`: Complete situational awareness `[batch, d_context]`
- `exteroceptive_context`: External perception `[batch, d_attn]`
- `interoceptive_context`: Internal state `[batch, d_attn]`
- `memory_context`: Retrieved memories `[batch, d_memory]`

## Training

### Loss Functions

```python
# 1. Predictive Loss (world model)
predicted_next_state = forward_model(situation_context, action)
loss_pred = F.mse_loss(predicted_next_state, actual_next_state)

# 2. Intrinsic Reward (homeostasis)
reward_intrinsic = -homeostasis_violation(homeostasis)

# 3. Policy Gradient (reinforcement learning)
loss_pg = -torch.log(action_probs.gather(1, action)) * reward_intrinsic

# Total loss
loss = loss_pred + lambda_pg * loss_pg
```

### Training Loop

```python
optimizer = torch.optim.Adam(vem.parameters(), lr=1e-4)

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    for t in range(max_timesteps):
        # Get observations
        visual, audio, proprio, homeo = state
        
        # Forward pass
        output = vem(visual, audio, proprio, homeo)
        
        # Sample action
        action_dist = torch.distributions.Categorical(output['action_probs'])
        action = action_dist.sample()
        
        # Environment step
        next_state, reward, done = env.step(action)
        
        # Compute loss (example: policy gradient)
        loss = -action_dist.log_prob(action) * reward
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update memory
        action_onehot = F.one_hot(action, vem.n_actions).float()
        vem.update_memory(output['situation_context'], action_onehot)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
```

## Validation

### Unit Tests

Run unit tests to verify implementation:

```bash
python tests/test_attention_mechanisms.py
python tests/test_memory.py
python tests/test_forward_pass.py
```

### Gradient Flow Check

```python
# Verify gradients flow through all components
output = vem(visual, audio, proprio, homeo)
loss = output['action_logits'].sum()
loss.backward()

# Check all parameters have gradients
for name, param in vem.named_parameters():
    if param.grad is None:
        print(f"WARNING: {name} has no gradient")
    else:
        print(f"✓ {name}: grad_norm = {param.grad.norm().item():.4f}")
```

## Integration with Simulation Environments

### Compatible Environments

The VEM can interface with:

1. **MuJoCo** - High-fidelity physics simulation
2. **PyBullet** - Robot simulation
3. **Isaac Gym** - GPU-accelerated physics
4. **Custom GridWorld** - Simplified 2D environments

### Example: GridWorld Integration

```python
class GridWorldWrapper:
    def __init__(self, grid_size=8):
        self.env = GridWorld(grid_size)
        
    def get_observations(self):
        # Convert grid to visual features
        visual = self.env.render_first_person()  # [h, w, 3]
        visual_features = cnn_encoder(visual)     # [n_regions, d_v]
        
        # Audio (e.g., nearby object sounds)
        audio = self.env.get_audio_features()     # [n_bins, d_a]
        
        # Proprioception (position, orientation)
        proprio = torch.tensor([
            self.env.agent_x, 
            self.env.agent_y,
            self.env.agent_orientation,
            # ... other kinematic info
        ])
        
        # Homeostasis (energy, damage)
        homeo = torch.tensor([
            self.env.agent_energy,
            self.env.agent_damage,
            # ... other internal states
        ])
        
        return visual_features, audio, proprio, homeo
```

## Reproducibility

### Random Seed

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
```

### Deterministic Behavior

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Save/Load Models

```python
# Save
torch.save({
    'model_state_dict': vem.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': model_config,
}, 'vem_checkpoint.pt')

# Load
checkpoint = torch.load('vem_checkpoint.pt')
vem.load_state_dict(checkpoint['model_state_dict'])
```

## Performance Benchmarks

### Forward Pass Speed (CPU)

| Batch Size | Time (ms) | Throughput (samples/sec) |
|------------|-----------|--------------------------|
| 1          | ~15       | ~67                      |
| 4          | ~25       | ~160                     |
| 16         | ~80       | ~200                     |
| 32         | ~150      | ~213                     |

*Measured on Intel i7-9700K @ 3.60GHz*

### Memory Usage

| Configuration | Parameters | Memory (MB) |
|---------------|------------|-------------|
| Small         | ~500K      | ~200        |
| Default       | ~2.5M      | ~500        |
| Large         | ~10M       | ~1500       |

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{quiroga2025body,
  title={The Body is the Key: A Formal Architecture for Embodied Grounding in Artificial General Intelligence},
  author={Quiroga Rodríguez, Elio},
  journal={Array},
  year={2025},
  note={Under Review}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

- **Author:** Elio Quiroga Rodríguez
- **Email:** elio.quiroga@pdi.atlanticomedio.es
- **Institution:** Universidad del Atlántico Medio

## Acknowledgments

This implementation builds upon the transformer architecture introduced by Vaswani et al. (2017) and extends it to the domain of embodied cognition, drawing inspiration from ecological psychology (Gibson, 1979), enactive cognitive science (Varela et al., 1991), and developmental robotics (Pfeifer & Bongard, 2006).
