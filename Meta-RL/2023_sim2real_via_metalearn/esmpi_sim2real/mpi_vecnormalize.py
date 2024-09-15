import gymnasium as gym
import numpy as np

from mpi4py import MPI

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd




class MPIVecNormalize(VecNormalize):
    def sync_statistics(self):
        """
        Broadcast the statistics from the master's env statistics to the workers.
        The master is the process with MPI rank 0.
        """

        # collect stats from all workers first
        env_stats = (self.obs_rms, self.ret_rms)
        env_stats = MPI.COMM_WORLD.gather(env_stats, root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            for i in range(1, len(env_stats)):
                self.obs_rms.combine(env_stats[i][0])
                self.ret_rms.combine(env_stats[i][1])

        env_stats = (self.obs_rms, self.ret_rms)
        env_stats = MPI.COMM_WORLD.bcast(env_stats, root=0)
        self.obs_rms, self.ret_rms = env_stats


"""
def test_sync_statistics():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = make_vec_env('CartPole-v1',
                    n_envs=4,
                    vec_env_cls=DummyVecEnv)
    env = MPIVecNormalize(env)


    if rank == 0:
        # Modify obs_rms and ret_rms
        env.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        env.ret_rms = RunningMeanStd(shape=())
    env.ret_rms.mean = rank+0.1

    print('\033[92m before sync: ',rank, env.ret_rms.mean, '\033[0m')

    env.sync_statistics()

    print('\033[92m FINALE: ',rank, env.ret_rms.mean, '\033[0m')

test_sync_statistics()
#"""