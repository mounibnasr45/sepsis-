import os
import torch
import torch.nn as nn
from torch.distributions import Categorical
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# --- Load the trained PPO agent ---
def load_ppo_agent(model_path, state_dim, action_dim):
    """Loads the trained PPO agent's actor network."""
    class ActorCritic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(ActorCritic, self).__init__()
            self.shared = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            self.actor = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Softmax(dim=-1)
            )
            self.critic = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, state):
            shared_features = self.shared(state)
            action_probs = self.actor(shared_features)
            state_value = self.critic(shared_features)
            return action_probs, state_value

        def act(self, state, memory=None):
            action_probs, _ = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            if memory is not None:
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(dist.log_prob(action))
            return action.item()

        def evaluate(self, state, action):
            action_probs, state_value = self.forward(state)
            dist = Categorical(action_probs)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            return action_logprobs, state_value, dist_entropy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ActorCritic(state_dim, action_dim).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))  # Load to the correct device
    agent.eval()  # Set to evaluation mode
    return agent, device

# --- Basic Simulation Function (Replace with your actual environment interaction) ---
import numpy as np

def simulate_step(current_state, action):
    """
    Simulates the patient's next state with a tendency to worsen.
    """
    next_state = list(current_state)
    reward = 0

    # Define probabilities for worsening trends
    prob_worsen = 0.2
    worsening_factor = 0.95  # Multiply by this to decrease (or increase for temp/WBC)

    # Simulate changes in vital signs and labs with a bias towards worsening
    next_state[0] += np.random.normal(0, 5) * (1 if np.random.rand() > prob_worsen else -1)  # HR
    next_state[1] += np.random.normal(0, 5) * (1 if np.random.rand() > prob_worsen else -1)  # SysBP
    next_state[2] += np.random.normal(0, 3) * (1 if np.random.rand() > prob_worsen else -1)  # DiaBP
    next_state[3] += np.random.normal(0, 1) * (1 if np.random.rand() > prob_worsen else -1)  # RR
    next_state[4] += np.random.normal(0, 0.1) * (1 if np.random.rand() > prob_worsen else -1)  # Temp
    next_state[5] += np.random.normal(0, 1) * (1 if np.random.rand() > prob_worsen else -1)  # O2
    next_state[6] *= (worsening_factor if np.random.rand() < prob_worsen else 1 / worsening_factor + np.random.normal(0, 0.05))  # Lactate (increase)
    next_state[7] *= (worsening_factor if np.random.rand() < prob_worsen else 1 / worsening_factor + np.random.normal(0, 0.05))  # WBC (can vary)

    # Ensure values stay within reasonable bounds (very basic)
    next_state[0] = max(30, min(180, next_state[0]))  # HR
    next_state[1] = max(50, min(220, next_state[1]))  # SysBP
    next_state[2] = max(30, min(120, next_state[2]))  # DiaBP
    next_state[3] = max(5, min(40, next_state[3]))  # RR
    next_state[5] = max(70, min(100, next_state[5]))  # O2
    next_state[6] = max(0.5, min(15, next_state[6]))  # Lactate
    next_state[7] = max(1, min(30, next_state[7]))  # WBC

    # Update time features
    next_state[8] += 1
    next_state[9] += 1

    # Reward based on a simplified "worsening" condition
    if next_state[6] > current_state[6] + 0.5 or next_state[0] > current_state[0] + 10 or next_state[1] < current_state[1] - 10:
        reward = -0.3  # Small negative reward for deterioration

    return next_state, reward

# --- Action Mapping ---
action_map = {
    0: "Do nothing",
    1: "Administer IV fluids",
    2: "Administer antibiotics",
    3: "Administer vasopressors",
    4: "Alert clinician",
    5: "Combo (fluids + antibiotics)"
}

# --- Initial Patient State ---
initial_state = [100, 95, 65, 18, 37.0, 98, 1.5, 8, 0, 0]

if 'current_state' not in st.session_state:
    st.session_state['current_state'] = initial_state
if 'history' not in st.session_state:
    st.session_state['history'] = [initial_state]
if 'actions_taken' not in st.session_state:
    st.session_state['actions_taken'] = []
if 'rewards_received' not in st.session_state:
    st.session_state['rewards_received'] = []
if 'step_number' not in st.session_state:
    st.session_state['step_number'] = 0

st.title("RL Agent for Early Sepsis Detection & Intervention Simulation")

# Load the PPO agent
MODEL_PATH = "sepsis_ppo_model_final.pth"  # Replace with the actual path to your trained model
STATE_DIM = 10  # Dimension of your state
ACTION_DIM = 6  # Number of possible actions
ppo_agent, device = load_ppo_agent(MODEL_PATH, STATE_DIM, ACTION_DIM)

placeholder = st.empty()  # Create a placeholder for updating the entire display

while True:
    with placeholder.container():
        current_state = st.session_state['current_state']
        history = st.session_state['history']
        actions_taken = st.session_state['actions_taken']
        rewards_received = st.session_state['rewards_received']
        step_number = st.session_state['step_number']

        st.subheader(f"Patient State - Step: {step_number}")
        state_labels = ["HR", "SysBP", "DiaBP", "RR", "Temp", "O2", "Lactate", "WBC", "Time Since Admission",
                        "Time Since Last Action"]
        state_df = pd.DataFrame([current_state], columns=state_labels)
        st.write(state_df)

        # --- Visualize Vitals ---
        st.subheader("Vitals Over Time")
        time_steps = np.arange(len(history))
        history_df = pd.DataFrame(history, columns=state_labels)

        fig_vitals, ax_vitals = plt.subplots(figsize=(10, 6))
        for i, col in enumerate(
                ["HR", "SysBP", "DiaBP", "RR", "Temp", "O2", "Lactate", "WBC"]):
            ax_vitals.plot(time_steps, history_df[col], label=col)
        ax_vitals.set_xlabel("Time Step")
        ax_vitals.set_ylabel("Value")
        ax_vitals.legend()
        st.pyplot(fig_vitals)

        # --- Show Agent's Action ---
        if actions_taken:
            last_action = actions_taken[-1]
            st.subheader("Agent's Last Action:")
            st.info(f"Step {step_number}: {last_action}")

        # --- Show Rewards ---
        if rewards_received:
            st.subheader("Rewards Received:")
            rewards_df = pd.DataFrame({'Step': range(1, len(rewards_received) + 1), 'Reward': rewards_received})
            st.write(rewards_df)

            fig_rewards, ax_rewards = plt.subplots()
            ax_rewards.plot(rewards_df['Step'], rewards_df['Reward'])
            ax_rewards.set_xlabel("Step")
            ax_rewards.set_ylabel("Reward")
            st.pyplot(fig_rewards)

        # --- Agent's Action Selection (Replace with your RL agent's logic) ---
        state_tensor = torch.FloatTensor(current_state).to(device).unsqueeze(0)
        action_probs, _ = ppo_agent(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample().item()

        st.session_state['actions_taken'].append(action_map[action])

        next_state, reward = simulate_step(current_state, action)
        st.session_state['history'].append(next_state)
        st.session_state['rewards_received'].append(reward)
        st.session_state['current_state'] = next_state
        st.session_state['step_number'] += 1

        time.sleep(2)
