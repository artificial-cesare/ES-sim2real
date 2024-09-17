"""
Run as:
    mpirun -n (n_workers) python example_es.py

Important: if using a shared GPU on the same node, you should also run:
    # nvidia-cuda-mps-control -d

"""

import time
import os
import argparse
import shutil

import torch
import torch.nn as nn

import gymnasium as gym

from mpi4py import MPI

from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, is_vecenv_wrapped, VecMonitor, VecNormalize
from stable_baselines3 import PPO

from sb3_contrib import RecurrentPPO

from esmpi_sim2real.esmpi import ESMPI, compute_centered_ranks
from esmpi_sim2real.mpi_vecnormalize import MPIVecNormalize
from esmpi_sim2real.hopper_wrappers import MLRandomizedEnvWrapper
from esmpi_sim2real.algorithms import adapt_and_evaluate_OpenAIES
from esmpi_sim2real.dr_wrapper import DR_Wrapper, Auto_DR_Wrapper


parser = argparse.ArgumentParser(
                    prog='ES-MPI',
                    description='',
                    epilog='')

parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--env', default='Hopper-v4', help="env name")
parser.add_argument('--outdir', default='out', help="directory to save results to")

args = parser.parse_args()
random_seed = args.seed
outdir = args.outdir
envname = args.env
print('outdir: ', outdir)
print('envname: ', envname)
print('seed: ', random_seed)

# Save params to the output directory as params.txt

if not os.path.exists(outdir):
    os.makedirs(outdir)
with open(os.path.join(outdir, 'params.txt'), 'w') as f:
    f.write(str(args))
# Copy this whole script to the output directory
shutil.copy(__file__, outdir)

policy_kwargs = dict(
                log_std_init=-2,
                activation_fn=nn.Tanh,
                net_arch=dict(pi=[256], vf=[256]),
                lstm_hidden_size=256,
                )

if envname == 'Hopper-v4':
    dr_specs_hopper = {"body('torso').mass": {"uniform": [0.35, 9.75], "type" : "="},
                        "body('thigh').mass": {"uniform": [0.35, 9.75], "type" : "="},
                        "body('leg').mass": {"uniform": [0.35, 9.75], "type" : "="},
                        "body('foot').mass": {"uniform": [0.35, 9.75], "type" : "="},
                        "joint('foot_joint').damping" : {"uniform": [0.17, 2.93], "type" : "="},
                        "joint('leg_joint').damping" : {"uniform": [0.17, 2.93], "type" : "="},
                        "joint('thigh_joint').damping" : {"uniform": [0.17, 2.93], "type" : "="},
                        "geom('foot_geom').friction[0]": {"uniform": [0.17, 2.93], "type" : "="}}
else:
    print('env not supported')


device = torch.device("cuda")


# seed + mpi4py rank id
seed = random_seed
set_random_seed(seed, using_cuda=torch.cuda.is_available())


if __name__ == '__main__':
    def make_env(rank, seed, wrapper_class):
        def _init():
            env = gym.make(envname)#, healthy_reward=0, reset_noise_scale=5e-2)
            env = wrapper_class(env, randomization_specs=dr_specs_hopper)
            env.reset(seed=seed+rank)
            return env
        return _init 
    
    env = DummyVecEnv([make_env(i, seed) for i in range(16)]) #
    env = VecMonitor(env)

    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env.reset()


    model = RecurrentPPO("MlpLstmPolicy",
                env, verbose=True,
                policy_kwargs=policy_kwargs,
                n_steps=128,
                batch_size=256,
                n_epochs=3,
                gamma=0.995,
                tensorboard_log='tensorboard',
                sde_sample_freq=4, # check, depending on mujoco env
                use_sde=True)
    
    print(model.policy)

    model.learn(3000000)
