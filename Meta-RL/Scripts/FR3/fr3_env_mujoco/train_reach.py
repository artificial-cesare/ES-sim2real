import time
import numpy as np
import os

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback


from fr3_tasks.task_fr3_reach import FR3Reach

def make_env(rank, seed=0):
    def _init():
        env = FR3Reach(render_mode=None)

        env.reset(seed = seed + rank)
        return env
    return _init



if __name__ == "__main__":
    """
    env = SubprocVecEnv([make_env(i) for i in range(16)])

    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., norm_obs_keys=["observation"])

    # TODO: best to set policy_kwargs so we know exactly what network architecture we are using

    model = PPO("MultiInputPolicy", env, verbose=1, n_steps=256, batch_size=256, n_epochs=3, gamma=0.995, tensorboard_log="tensorboard/")

    print(model.policy)


    model.learn(1_000_000)

    model.save(os.path.join("tmp_out", "fizzy"))
    env.save(os.path.join("tmp_out", "vecnormalize.pkl"))
    #"""




    model = PPO.load(os.path.join("tmp_out", "fizzy"))

    render_mode='human'
    #render_mode=None
    env = DummyVecEnv([lambda:FR3Reach(render_mode=render_mode)])
    env = VecNormalize.load(os.path.join("tmp_out", "vecnormalize.pkl"), env)

    env.training = False
    env.norm_reward = False

    for i in range(10):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.5)


## TODO: the robot is moving too fast; 300ms (3 env steps) to target is ridiculous; work on params, gains, speed limits
