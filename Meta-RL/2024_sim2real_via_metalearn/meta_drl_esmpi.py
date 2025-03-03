"""
Run as:
    mpirun -n (n_workers) python example_es.py
    mpirun --oversubscribe -n 8 python3 meta_drl_esmpi.py -> 8 workers

Important: if using a shared GPU on the same node, you should also run:
    # nvidia-cuda-mps-control -d
"""

import time

import torch
import torch as th
import torchvision
import numpy as np
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mpi4py import MPI

from gymnasium.wrappers import TimeLimit

from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, is_vecenv_wrapped, VecMonitor

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO


from esmpi_sim2real.esmpi import ESMPI, compute_centered_ranks, compute_identity_ranks
from esmpi_sim2real.metarl_envs import HalfCheetahDirEnv, HalfCheetahVelEnv
from esmpi_sim2real.mpi_vecnormalize import MPIVecNormalize

from esmpi_sim2real.hopper_mpi import MLRandomizedEnvWrapper, MLAutoRandomizedEnvWrapper

# TODO: to fine-tune centered-ranks vs identity-ranks, lr, pop size (e.g., in prev paper they used 300)


# batch-size for the outer (meta) loop
population_size = 16*12 # make sure this is a multiple of the number of workers used

n_parallel_envs_per_worker = 1  # not strictly required, but it can be useful to speed-up
                                # the inner-loop operations; this is especially true for the
                                # evaluation of adapted policies


# TODO: it would be best to ppo.train 

num_inner_loop_ppo_iterations = 6

ppo_n_steps = 512
ppo_batch_size = 64
num_total_inner_loop_steps = num_inner_loop_ppo_iterations * (n_parallel_envs_per_worker * ppo_n_steps)
# TODO: check: num_total_inner_loop_steps should hopefully be a few episodes; but note: if using parallel environments, the total 
#       number of steps will be reached well before the end of an episode!
# TODO: auto balance all these parameters

print('PPO inner loop steps: ', num_total_inner_loop_steps)
if num_total_inner_loop_steps % ppo_batch_size != 0:
    print('\n\n*** WARNING: ',num_total_inner_loop_steps,'should be a multiple of ppo_batch_size ', ppo_batch_size, '\n\n')


outer_loop_lr = 0.003
inner_loop_lr = 3e-4 #0.003
es_sigma = 0.02

random_seed = 42

n_meta_iterations = 400



### [ CREATE ENV and PPO for each worker ]
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Using GPU')
else:
    device = torch.device("cpu")
    print('Using CPU')

is_master_worker = (MPI.COMM_WORLD.Get_rank()==0)

# seed + mpi4py rank id
seed = random_seed + MPI.COMM_WORLD.Get_rank()
set_random_seed(seed, using_cuda=torch.cuda.is_available())

"""
# TODO: for hopper, as in the planned work (Cesare)
policy_kwargs = dict(log_std_init=-2,
                        ortho_init=False,
                        activation_fn=nn.Tanh,
                        net_arch=dict(pi=[128, 128], vf=[128, 128])
                    )

env = make_vec_env('Hopper-v4',
                    n_envs=n_parallel_envs_per_worker,
                    seed=seed,
                    vec_env_cls=DummyVecEnv) #wrapper_class=ManuallyRandomizedEnvWrapper)

"""
# """
## Hopper-v4 

policy_kwargs = dict(log_std_init=-2,
                    ortho_init=False,
                    activation_fn=nn.ReLU,
                    net_arch=dict(pi=[128, 64], vf=[128, 64])
                    )
#seeds = [seed + i for i in range(n_parallel_envs_per_worker)]

env = make_vec_env('Hopper-v4', n_envs=n_parallel_envs_per_worker, seed=seed, vec_env_cls=DummyVecEnv, wrapper_class=MLRandomizedEnvWrapper)
randomize_fn = env.env_method('sample_task')
set_task_fn = env.get_attr('set_task')

for i in range(len(randomize_fn)):
    set_task_fn[i](randomize_fn[i])

## !!! could not work with vec_env since it's not a method of the vecenv, but of the envs inside it
env = TimeLimit(env, max_episode_steps=1000)
env = VecMonitor(env)
# """

"""
## HalfCheetahDirEnv
# watch out: must wrap with timelimit BEFORE it's wrapped with monitorenv!
policy_kwargs = dict(
                log_std_init=-2,
                activation_fn=nn.Tanh,
                net_arch=dict(pi=[128,128], vf=[128, 128])
                )

def make_env(rank, seed):
    def _init():
        env = HalfCheetahDirEnv()
        #env = HalfCheetahVelEnv()

        env = TimeLimit(env, max_episode_steps=1000)
        env.reset(seed=seed+rank)
        return env
    return _init
env = DummyVecEnv([make_env(i, seed) for i in range(n_parallel_envs_per_worker)])
env = VecMonitor(env)
"""

## TODO: make 2 envs;  one with vecmonitor after vecnormalize, making sure it is applied, for use in eval_fn;  
##       the other opposite, for a good evaluation after meta iterations
##       perhaps i can just use the vecnormalize unnormalized-rewards method 



# very important: with mujoco envs, we need to use the same env normalization statistics in all workers and envs!
#   must implement perhaps an 'mpi-shared-vecnormalize-wrapper';  kind of vecnormalize wrapper but across envs and workers (threads or processes/nodes)
# [# +++ perhaps, vecnormalize can keep per-worker statistics, and synchronize them only at the end of each outer-loop iteration;   explicit 'sync_statistics' method to the wrapper]

# TODO:
env = MPIVecNormalize(env, norm_obs=True, norm_reward=True)
env.reset()


# TODO: if/when using reward-function-metalearning, i will need another wrapper to anneal between the original env reward and the predicted one;  i will need 
#   one such wrapper anyway to replace the env reward with the predicted one


# Note: ppo optimizer for inner-loop steps; in particular,
# the number of ppo iterations performed is equal to 
# num_total_inner_loop_steps / (n_steps * n_parallel_envs_per_worker)

model_worker = PPO("MlpPolicy",
            env, verbose=False,
            #policy_kwargs=policy_kwargs,
            n_steps=ppo_n_steps,
            batch_size=ppo_batch_size,
            n_epochs=4,
            learning_rate=inner_loop_lr,
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            tensorboard_log=None,
            sde_sample_freq=-1, # check, depending on mujoco env
            use_sde=False)

model_results = PPO("MlpPolicy",
            env, verbose=False,
            #policy_kwargs=policy_kwargs,
            n_steps=ppo_n_steps,
            batch_size=ppo_batch_size,
            n_epochs=4,
            learning_rate=inner_loop_lr,
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            tensorboard_log=None,
            sde_sample_freq=-1, # check, depending on mujoco env
            use_sde=False)

# TODO: to try
"""
- running: no vecnormalize, no sde_sampling
- try without fitness rank-normalization
- try a run with much larger pop size
- implement ESReptile too, using mpi for parallel workers;  note that in that case i do need to send the final weights though = high bandwidth

- check/decide:  vecnormalize useful for rewards, but inner-loop/outer-loop possible discrepancy (i.e.,  evaluate_policy takes real rewards, but then fitnesses are rank-normalized anyway)
- try with longer inner-loop steps

Reuse recent perturbations and computed fitness values; if the weights change by little, then the prev weights+perturb may still be close enough; the distrivution however has changed,
so care must be taken to keep it normally distr
+++   Track: distance between metaparams at each iteration wrt first weights, to see how much i move away in param space, and see if it may be worth to never throw away 
perturbs (unless particularly far away / with goo small inp sampling weight)
   And also count the number of prev perturbed weights within 1,2,3 sigmas from the current weigts, at each iter, to gauge the possible impact.  1s=68% 2s=95%

        ---> if outer-loop-lr << es_sigma, then consecutive weights will move very little, and thus the prev perturbations will be almost the same distance from the new weights. similar to ES-IS paper?  Repeat update many times?

   
# sgd instead of adam


+ Note: inner-loop adptation smooths out the fitbess landscape; taking the final weights after adapt and faking them with the same final fitness (as a lower bound) may be useful, 
since they mqy be closer to the new weights, and thus have a higher imp. Sampling score
+++++ OR: use the final adapted weights INSTEAD of the initial perturbed ones, to speed up learning


[spin-off project:  imp-sampling reuse previous perturbations, like in the paper i found,  but also using the final weights as weight perturbs;  note: no longer normally distributed,
so exp. must be corrected!!]   ->  ES-MAML+




IMP?


still, es without inner-loop adaptation learns faster and is much faster to grow (much less variance)

+++++and the perturbated-weights final reward, AFTER adaptation, and worse (and at least with higher variance) than non-adapted weights,
but inner-loop adaptation should generally improve the performance of the agent



++ TODO: make it work with multitask
++ TODO: put back mpivecnormalize!


parallel envs to decorrelate ppo?


+++++ run on snellius, cpu vs gpu!


[keep in mind: using last-adapted-weights instead of initial ones
               reusing perturbations from previous steps]





larger net, tanh, no sde, 1 parallel env + ppo minor options;  low ppo lr



TODO: double check no leaking of weights from inner-loops

TODO: on cluster, even when 1 worker per process, eval_fn is called twice, taking 7s instead of ~3

[ps:  low lr and high number of workers usually help]

"""





### [ ES-MPI meta-optimization, incl. outer-loop and inner-loop ]

#TODO: make sure passing ppo.policy is sufficient to make my ESMPI implementation work

# TODO: in es-maml for sim2real paper they don't normalize fitnesses: try with 'compute_identity_ranks' instead of 'compute_centered_ranks'
#fitness_transform_fn = compute_identity_ranks
fitness_transform_fn = compute_centered_ranks
optimizer = ESMPI(model_results.policy, learning_rate=outer_loop_lr, sigma=es_sigma, population_size=population_size, fitness_transform_fn=fitness_transform_fn, device=device)



### [ OUTER LOOP ]
last_t = time.time()
for meta_iteration in range(1, n_meta_iterations + 1):
    """
        decide how to randomize and split across batch elements (e.g., indep. sampling of ES perturbations and domain randomization
    -or metarl tasks-, or for each randomization to sample a number of perturbations)

number of episodes -- best to deal with episode-level?),  -> if timestep based, make sure that the number of timesteps chosen guarantees a sufficient number of full episodes (guaranteeable by max timesteps / timelimit envs)
then meta-train-testing and calculation of meta-gradient(s)
then meta-optimization step (update the policy.parameters() using the meta-gradient(s) and the outer-loop optimizer)
    """


    ### [ INNER LOOP ]
    def eval_fn(policy_to_evaluate):
        global model

        ####
        st = time.time()

        # copy the new policy 'model' into ppo.policy
        model_worker.policy.load_state_dict(policy_to_evaluate.state_dict())

        # reset the (local/worker) ppo object;   reset the optimizer
        model_worker.policy.optimizer = model_worker.policy.optimizer_class(model_worker.policy.parameters(), lr=get_schedule_fn(model_worker.learning_rate)(1), **model_worker.policy.optimizer_kwargs)

        # reset (parallel) env by sampling a new randomization and applying it to all the eventual parallel envs for the same worker
        env = model_worker.get_env()

        random_task = env.env_method('sample_task', indices=[0])[0] # params randomization
        
        ####random_task = {'direction': 1} ## TODO: remove; quick test with a single task
        ###random_task = {'velocity': 1} ## TODO: remove; quick test with a single task
        
        env.env_method('set_task', random_task) # TODO: do we want to reset the env? in some cases (like domain rand?) it may be necessary,
                                                # but we would lose the de-synchronization between parallel envs
        env.reset() # TODO: check this

        # perform inner-loop adaptation; here, with PPO, but it could be done in many ways (e.g., hill climbing, as in ``Rapidly Adaptable Legged Robots via Evolutionary Meta-Learning'')
        model_worker.learn(num_total_inner_loop_steps)

        # TODO:
        # evaluate the policy (either with evaluate_policy, though possibly on just 1-2 envs to reduce overheard, or just for a fixed number of timesteps)
        # note: evaluate_policy requires a vecmonitor or monitor wrapper; any timelimit wrapper must be applied before wrapping with the monitor
        mean_reward, _ = evaluate_policy(model_worker, env, n_eval_episodes=2, deterministic=True)

        # loss is the negative of the fitness/returns, since the optimizer performs gradient descent, and we want to maximize the fitness
        loss = -mean_reward

        ####
        en = time.time()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('eval_fn t: ', en-st, mean_reward, random_task)

        return loss
    loss = optimizer.step(eval_fn)
    ###


    if is_vecenv_wrapped(env, MPIVecNormalize):
        env.sync_statistics()


    if meta_iteration % 1 == 0 and optimizer.is_master:
        # EVALUATE CURRENT BEST PARAMETERS:
        # TODO: make utility function that takes extra arguments: taskid, number of eval episodes
        ##rewards = -eval_fn(optimizer.model)

        print('\t Iteration ', meta_iteration, ' time: ', time.time()-last_t)
        print('Rewards: ', -loss)
        ##print('Evaluated rewards: ', rewards)
        last_t = time.time()



if optimizer.is_master:
    # save policy and env statistics
    #torch.save(network.state_dict(), 'model.pth')
    torch.save(optimizer.model.state_dict(), 'model.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')