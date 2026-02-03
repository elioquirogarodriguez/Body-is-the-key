"""
Virtual Embodiment Module (VEM) - Core Implementation

This module implements the Virtual Embodiment Module as described in:
"The Body is the Key: A Formal Architecture for Embodied Grounding in AGI"

The VEM integrates three specialized attention mechanisms:
1. Exteroceptive Attention (external sensory processing)
2. Interoceptive Attention (internal state monitoring)
3. Actuator Attention (action selection)

Author: Elio Quiroga Rodríguez
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism as used in Vaswani et al. (2017).
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor [batch, n_q, d_k]
            k: Key tensor [batch, n_k, d_k]
            v: Value tensor [batch, n_k, d_v]
            mask: Optional mask [batch, n_q, n_k]
        
        Returns:
            attended_values: [batch, n_q, d_v]
            attention_weights: [batch, n_q, n_k]
        """
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights


class ExteroceptiveAttention(nn.Module):
    """
    Processes external sensory inputs (vision, audition).
    
    Visual input is treated as a spatial grid of features.
    Auditory input is treated as time-frequency bins.
    """
    
    def __init__(self, d_visual: int, d_audio: int, d_attn: int):
        super().__init__()
        
        # Visual attention projections
        self.W_v_Q = nn.Linear(d_visual, d_attn)
        self.W_v_K = nn.Linear(d_visual, d_attn)
        self.W_v_V = nn.Linear(d_visual, d_attn)
        
        # Audio attention projections
        self.W_a_Q = nn.Linear(d_audio, d_attn)
        self.W_a_K = nn.Linear(d_audio, d_attn)
        self.W_a_V = nn.Linear(d_audio, d_attn)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_attn))
        
        # Fusion layer
        self.fusion = nn.Linear(2 * d_attn, d_attn)
        
        self.d_attn = d_attn
    
    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            visual: Visual features [batch, n_visual_regions, d_visual]
            audio: Audio features [batch, n_audio_bins, d_audio]
        
        Returns:
            context: Exteroceptive context vector [batch, d_attn]
            attention_weights: Dictionary of attention maps
        """
        # Visual attention
        Q_v = self.W_v_Q(visual)
        K_v = self.W_v_K(visual)
        V_v = self.W_v_V(visual)
        
        visual_attended, attn_v = self.attention(Q_v, K_v, V_v)
        visual_context = visual_attended.mean(dim=1)  # Pool over spatial regions
        
        # Auditory attention
        Q_a = self.W_a_Q(audio)
        K_a = self.W_a_K(audio)
        V_a = self.W_a_V(audio)
        
        audio_attended, attn_a = self.attention(Q_a, K_a, V_a)
        audio_context = audio_attended.mean(dim=1)  # Pool over time-frequency bins
        
        # Fuse visual and audio contexts
        fused = torch.cat([visual_context, audio_context], dim=-1)
        context = self.fusion(fused)
        
        attention_weights = {
            'visual': attn_v,
            'audio': attn_a
        }
        
        return context, attention_weights


class InteroceptiveAttention(nn.Module):
    """
    Monitors internal body state (proprioception, homeostasis).
    
    Attends to recent history of internal states to detect salient changes
    (e.g., pain signals, energy depletion).
    """
    
    def __init__(self, d_proprio: int, d_homeo: int, d_attn: int, 
                 memory_window: int = 10):
        super().__init__()
        
        d_internal = d_proprio + d_homeo
        
        # Internal state attention projections
        self.W_i_Q = nn.Linear(d_internal, d_attn)
        self.W_i_K = nn.Linear(d_internal, d_attn)
        self.W_i_V = nn.Linear(d_internal, d_attn)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_attn))
        
        self.memory_window = memory_window
        self.d_attn = d_attn
    
    def forward(self, proprioception: torch.Tensor, 
                homeostasis: torch.Tensor,
                internal_history: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            proprioception: Current proprioceptive state [batch, d_proprio]
            homeostasis: Current homeostatic state [batch, d_homeo]
            internal_history: Recent internal states [batch, window, d_internal]
        
        Returns:
            context: Interoceptive context vector [batch, d_attn]
            attention_weights: Attention over time window [batch, 1, window]
        """
        # Concatenate current internal state
        current_internal = torch.cat([proprioception, homeostasis], dim=-1)
        
        # If no history, create a dummy sequence with just current state
        if internal_history is None:
            internal_history = current_internal.unsqueeze(1)
        
        # Current state forms the query
        Q_i = self.W_i_Q(current_internal).unsqueeze(1)  # [batch, 1, d_attn]
        
        # History forms keys and values
        K_i = self.W_i_K(internal_history)  # [batch, window, d_attn]
        V_i = self.W_i_V(internal_history)  # [batch, window, d_attn]
        
        # Attend to history
        context, attn_weights = self.attention(Q_i, K_i, V_i)
        context = context.squeeze(1)  # [batch, d_attn]
        
        return context, attn_weights


class ActuatorAttention(nn.Module):
    """
    Selects actions based on attended perceptual and internal state.
    
    Treats action prototypes as keys to be matched against the current
    situational query, producing a policy distribution π(a|s).
    """
    
    def __init__(self, d_context: int, n_actions: int, d_action: int):
        super().__init__()
        
        # Learnable action key embeddings
        self.action_keys = nn.Parameter(torch.randn(n_actions, d_action))
        
        # Project situation to action query space
        self.situation_to_query = nn.Linear(d_context, d_action)
        
        # Temperature for action selection
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.n_actions = n_actions
        self.d_action = d_action
    
    def forward(self, situation_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            situation_context: Fused perceptual and internal context [batch, d_context]
        
        Returns:
            action_probs: Policy distribution π(a|s) [batch, n_actions]
            action_logits: Raw logits before softmax [batch, n_actions]
        """
        # Create query from current situation
        Q_act = self.situation_to_query(situation_context)  # [batch, d_action]
        
        # Compute similarity with all action keys
        # Q: [batch, d_action], K: [n_actions, d_action]
        action_logits = torch.matmul(Q_act, self.action_keys.T)  # [batch, n_actions]
        action_logits = action_logits / self.temperature
        
        # Convert to probability distribution
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, action_logits


class EpisodicMemory(nn.Module):
    """
    Differentiable episodic memory for maintaining world state continuity.
    
    Stores attended experiences and retrieves relevant memories based on
    current context.
    """
    
    def __init__(self, d_memory: int, memory_slots: int):
        super().__init__()
        
        # Memory matrix M
        self.memory = nn.Parameter(torch.randn(memory_slots, d_memory))
        
        # Encoding network
        self.encoder = nn.Sequential(
            nn.Linear(d_memory, d_memory),
            nn.ReLU(),
            nn.Linear(d_memory, d_memory)
        )
        
        # Attention for reading
        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_memory))
        
        self.memory_slots = memory_slots
        self.d_memory = d_memory
    
    def write(self, experience: torch.Tensor, write_strength: float = 0.1):
        """
        Write new experience to memory (simplified LRU-like mechanism).
        
        Args:
            experience: Encoded experience [batch, d_memory]
            write_strength: How much to update memory
        """
        # Encode the experience
        encoded = self.encoder(experience)
        
        # For simplicity, write to first slot (in practice, use attention-based addressing)
        with torch.no_grad():
            self.memory.data[0] = (1 - write_strength) * self.memory.data[0] + \
                                   write_strength * encoded.mean(dim=0)
    
    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant memories based on current context.
        
        Args:
            query: Current context [batch, d_memory]
        
        Returns:
            memory_context: Retrieved memory [batch, d_memory]
            attention_weights: Attention over memory slots [batch, 1, memory_slots]
        """
        query = query.unsqueeze(1)  # [batch, 1, d_memory]
        memory_expanded = self.memory.unsqueeze(0).expand(query.size(0), -1, -1)
        
        # Attend to memory
        memory_context, attn_weights = self.attention(
            query, memory_expanded, memory_expanded
        )
        
        memory_context = memory_context.squeeze(1)
        
        return memory_context, attn_weights


class VirtualEmbodimentModule(nn.Module):
    """
    Complete Virtual Embodiment Module integrating all attention mechanisms.
    
    This is the main architecture that processes multi-modal sensory input,
    maintains internal state awareness, retrieves relevant memories, and
    selects actions in a continuous perception-action loop.
    """
    
    def __init__(self, 
                 d_visual: int = 512,
                 d_audio: int = 128,
                 d_proprio: int = 64,
                 d_homeo: int = 32,
                 d_attn: int = 256,
                 n_actions: int = 10,
                 d_action: int = 128,
                 memory_slots: int = 100):
        super().__init__()
        
        # Three specialized attention mechanisms
        self.exteroceptive = ExteroceptiveAttention(d_visual, d_audio, d_attn)
        self.interoceptive = InteroceptiveAttention(d_proprio, d_homeo, d_attn)
        self.actuator = ActuatorAttention(2 * d_attn + d_attn, n_actions, d_action)
        
        # Episodic memory
        d_memory = d_attn * 2 + d_action
        self.memory = EpisodicMemory(d_memory, memory_slots)
        
        # Experience encoder (for writing to memory)
        self.experience_encoder = nn.Linear(2 * d_attn + n_actions, d_memory)
        
        self.d_attn = d_attn
        self.n_actions = n_actions
    
    def forward(self, 
                visual: torch.Tensor,
                audio: torch.Tensor,
                proprioception: torch.Tensor,
                homeostasis: torch.Tensor,
                internal_history: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> dict:
        """
        Complete forward pass through VEM.
        
        Args:
            visual: Visual input [batch, n_visual_regions, d_visual]
            audio: Audio input [batch, n_audio_bins, d_audio]
            proprioception: Proprioceptive state [batch, d_proprio]
            homeostasis: Homeostatic state [batch, d_homeo]
            internal_history: Optional history [batch, window, d_internal]
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing:
                - action_probs: Policy distribution [batch, n_actions]
                - action_logits: Action logits [batch, n_actions]
                - context: Complete situation context [batch, d_context]
                - attention_maps: (optional) All attention weights
        """
        # Process external senses
        extero_context, extero_attn = self.exteroceptive(visual, audio)
        
        # Process internal state
        intero_context, intero_attn = self.interoceptive(
            proprioception, homeostasis, internal_history
        )
        
        # Fuse perceptual contexts
        perceptual_context = torch.cat([extero_context, intero_context], dim=-1)
        
        # Retrieve relevant memories
        memory_context, memory_attn = self.memory.read(perceptual_context)
        
        # Create complete situation awareness
        situation_context = torch.cat([perceptual_context, memory_context], dim=-1)
        
        # Select action
        action_probs, action_logits = self.actuator(situation_context)
        
        # Prepare output
        output = {
            'action_probs': action_probs,
            'action_logits': action_logits,
            'situation_context': situation_context,
            'exteroceptive_context': extero_context,
            'interoceptive_context': intero_context,
            'memory_context': memory_context
        }
        
        if return_attention:
            output['attention_maps'] = {
                'exteroceptive': extero_attn,
                'interoceptive': intero_attn,
                'memory': memory_attn
            }
        
        return output
    
    def update_memory(self, situation_context: torch.Tensor, action: torch.Tensor):
        """
        Write current experience to episodic memory.
        
        Args:
            situation_context: Current situation [batch, d_context]
            action: One-hot action taken [batch, n_actions]
        """
        # Encode experience
        experience = torch.cat([situation_context, action], dim=-1)
        encoded_experience = self.experience_encoder(experience)
        
        # Write to memory
        self.memory.write(encoded_experience)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Virtual Embodiment Module (VEM) - Reference Implementation")
    print("=" * 70)
    
    # Define dimensions
    batch_size = 4
    n_visual_regions = 64  # e.g., 8x8 grid
    n_audio_bins = 32
    
    # Create model
    vem = VirtualEmbodimentModule(
        d_visual=512,
        d_audio=128,
        d_proprio=64,
        d_homeo=32,
        d_attn=256,
        n_actions=10,
        d_action=128,
        memory_slots=100
    )
    
    print(f"\nModel Parameters: {sum(p.numel() for p in vem.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in vem.parameters() if p.requires_grad):,}")
    
    # Create dummy sensory input
    visual = torch.randn(batch_size, n_visual_regions, 512)
    audio = torch.randn(batch_size, n_audio_bins, 128)
    proprioception = torch.randn(batch_size, 64)
    homeostasis = torch.randn(batch_size, 32)
    internal_history = torch.randn(batch_size, 10, 96)  # 10 timestep history
    
    print("\nInput Shapes:")
    print(f"  Visual: {visual.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Proprioception: {proprioception.shape}")
    print(f"  Homeostasis: {homeostasis.shape}")
    print(f"  Internal History: {internal_history.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = vem(visual, audio, proprioception, homeostasis, 
                     internal_history, return_attention=True)
    
    print("\nOutput Shapes:")
    print(f"  Action Probabilities: {output['action_probs'].shape}")
    print(f"  Situation Context: {output['situation_context'].shape}")
    print(f"  Exteroceptive Context: {output['exteroceptive_context'].shape}")
    print(f"  Interoceptive Context: {output['interoceptive_context'].shape}")
    
    print("\nAttention Maps:")
    print(f"  Visual Attention: {output['attention_maps']['exteroceptive']['visual'].shape}")
    print(f"  Audio Attention: {output['attention_maps']['exteroceptive']['audio'].shape}")
    print(f"  Interoceptive Attention: {output['attention_maps']['interoceptive'].shape}")
    print(f"  Memory Attention: {output['attention_maps']['memory'].shape}")
    
    # Sample action
    action_dist = torch.distributions.Categorical(output['action_probs'])
    action = action_dist.sample()
    action_onehot = F.one_hot(action, num_classes=10).float()
    
    print(f"\nSampled Actions: {action}")
    print(f"Action Distribution (first sample): {output['action_probs'][0].numpy()}")
    
    # Update memory
    vem.update_memory(output['situation_context'], action_onehot)
    print("\nMemory updated with current experience")
    
    print("\n" + "=" * 70)
    print("VEM forward pass successful!")
    print("=" * 70)
