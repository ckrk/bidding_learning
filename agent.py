import os

#from stable_baselines3 import DDPG

import torch
import gym
from agent_ddpg import agent_ddpg

from util import Client

# in Terminal anaconda
#python D:\GitHub\netzerotc\rangl\server.py

ENV_ID = "nztc-open-loop-v0"
MODEL_PATH = "saved_models/MODEL_open_loop_0"

# when running the agent locally, assume that the environment is accesible at localhost:5000
# when running a containerised agent, assume that the environment is accesible at $RANGL_ENVIRONMENT_URL (typically http://nztc:5000)
remote_base = os.getenv("RANGL_ENVIRONMENT_URL", "http://localhost:5000/")

client = Client(remote_base)

seed = int(os.getenv("RANGL_SEED", 123456))
instance_id = client.env_create(ENV_ID, seed)


client.env_monitor_start(
    instance_id,
    directory=f"monitor/{instance_id}",
    force=True,
    resume=False,
    video_callable=False,
)

obs = client.env_reset(instance_id)

# Get gym-environment to set up the agent properly
env = gym.make("rangl:nztc-open-loop-v0")

# Hyper parameter Settings for setting up the agent
BATCH_SIZE = 128
ACTOR_LR =1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99

# set up agent and load trained actor weights
trained_agent_dict = torch.load('trained_agent.pt')
agent = agent_ddpg(env, hidden_size=[400, 300], actor_learning_rate=ACTOR_LR, critic_learning_rate=CRITIC_LR, gamma=GAMMA, tau=1e-3, max_memory_size=50000, norm = 'none')
agent.actor.load_state_dict(trained_agent_dict)

   
while True:
    action = agent.get_action_rangl(obs)
    action = [float(action[0]), float(action[1]), float(action[2])]
    obs, reward, done, info = client.env_step(instance_id, action)
    print(instance_id, reward)
    if done:
        print(instance_id)
        break

client.env_monitor_close(instance_id)

print("done", done)


# make sure you print the instance_id as the last line in the script
print(instance_id)
