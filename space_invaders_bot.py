import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Create and preprocess the Atari environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

# Define and train the PPO model
model = PPO(
    "CnnPolicy",  # Use a convolutional neural network policy
    env,
    verbose=1,
    tensorboard_log="./ppo_space_invaders_tensorboard/",
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    learning_rate=2.5e-4,
    clip_range=0.1
)

# Train the model
model.learn(total_timesteps=1_000_000)

# Test the trained model
def test_trained_model():
    test_env = make_atari_env(env_id, n_envs=1, seed=42)
    test_env = VecFrameStack(test_env, n_stack=4)

    obs = test_env.reset()
    for _ in range(10_000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = test_env.step(action)
        test_env.render()

    test_env.close()

test_trained_model()
