import torch
import torch.nn as nn
import copy
import numpy as np

from mpi4py import MPI

from .optimizers import SGD, Adam


# Adapted from OpenAI - Evolution Strategies
def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def compute_identity_ranks(x):
    return x


class SharedNoiseTable(object):
    def __init__(self):
        seed = 123
        count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        print('Sampling {}M random numbers with seed {}'.format(count//1e6, seed))

        comm = MPI.COMM_WORLD
        shared_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

        float_size = MPI.FLOAT.Get_size()
        if shared_comm.Get_rank() == 0:
            nbytes = count * float_size
        else:
            nbytes = 0
        self._shared_mem = MPI.Win.Allocate_shared(nbytes, float_size, comm=shared_comm)

        self.buf, itemsize = self._shared_mem.Shared_query(0)
        assert itemsize == MPI.FLOAT.Get_size() 
        self.noise = np.ndarray(buffer=self.buf, dtype=np.float32, shape=(count,)) 
        assert self.noise.dtype == np.float32

        if shared_comm.Get_rank() == 0:
            self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
        comm.Barrier()

        print('Sampled {}M bytes'.format((self.noise.size*4)//1e6))

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)
# END from OpenAI - Evolution Strategies



class ESMPI():
    def __init__(self, model, learning_rate=0.001, sigma=0.1, population_size=160, use_antithetic_sampling=True, fitness_transform_fn=compute_centered_ranks, device=torch.device('cpu')):
        """
            population_per_worker: number of random noises to evaluate for each worker. Evaluation is performed serially.
        """

        self.model = model
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.population_size = population_size
        self.use_antithetic_sampling = use_antithetic_sampling
        self.device = device

        self.fitness_transform_fn = fitness_transform_fn

        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self.num_workers = self._comm.Get_size()
        self.is_master = (self._rank == 0)

        self.noise = SharedNoiseTable()
        self.rs = np.random.RandomState()

        self._optimizer = Adam(self.model, lr=self.learning_rate) #SGD(self.model, lr=self.learning_rate, momentum=0)
        #self._optimizer = SGD(self.model, lr=self.learning_rate)

        self.population_per_worker = self.population_size // self.num_workers
        if self.population_size % self.num_workers != 0:
            print("ERROR: population size ", self.population_size, " is not a multiple of the number of workers ", self.num_workers)
        if self.is_master:
            print("Running ", self.population_per_worker, " perturbations on each worker.")

        # Receive current base weights from the master
        initial_parameters = nn.utils.parameters_to_vector(self.model.parameters()).cpu().detach()
        initial_parameters = self._comm.bcast(initial_parameters, root=0)
        nn.utils.vector_to_parameters(initial_parameters.to(self.device), self.model.parameters())

        # Let's make a copy of the base parameters, since we will routinely replace them with perturbed ones.
        self.current_parameters = copy.deepcopy(initial_parameters)

        ### DELETE
        self.pickle_logs = {'fitness_values':[], 'weights':[], 'perturbed_weights':[], 'iter_loss':[]}


    def step(self, eval_fn):
        """
            eval_fn: function that assesses the performance of a model passed as argument. The function is indepdent, and can rely on global variables (e.g., for datasets or RL environments)

            Returns average fitness values for each worker (all fitnesses for master).
        """

        # Each worker samples its 'self.population_per_worker' random perturbations + evaluate their fitness
        worker_results = []
        for i in range(self.population_per_worker):
            #perturbation_i = np.random.randn(*self.current_parameters.shape).astype(np.float32)
            noise_idx = self.noise.sample_index(self.rs, len(self.current_parameters))
            perturbation_i = self.noise.get(noise_idx, len(self.current_parameters))

            fitness_values = []
            for j in range(2 if self.use_antithetic_sampling else 1):
                sign = 1 if j == 0 else -1
                parameters = self.current_parameters + sign * self.sigma * perturbation_i
                nn.utils.vector_to_parameters(parameters.to(self.device), self.model.parameters())

                fitness_values.append( eval_fn(self.model) )

            worker_results.append([noise_idx, fitness_values])

        # Workers send the perturbed weights (indices in the shared noise table) + the corresponding fitnesses to the master
        worker_results = self._comm.gather(worker_results, root=0)

        # The master computes the new weights and sends them to all workers -> or no master, and each worker send their data to all other workers
        if self.is_master:
            # Master
            worker_results = sum(worker_results, [])

            noise_ids, fitness_values = zip(*worker_results)


            # Fitness shaping
            all_fitnesses = np.asarray(fitness_values) # incl. antithetic pairs
            all_fitnesses_flattened = np.reshape(all_fitnesses, -1)
            fitnesses_ = compute_centered_ranks(np.asarray(all_fitnesses_flattened))
            fitnesses_ = np.reshape(fitnesses_, all_fitnesses.shape)

            iter_loss = np.mean(all_fitnesses_flattened)

            fitnesses = []
            for i in range(len(fitnesses_)):
            # (if antithetic, only save a single (perturbation, (perf(+epsilon)-perf(-epsilon))/2 ) )
                if self.use_antithetic_sampling:
                    fitnesses.append( fitnesses_[i,0] - fitnesses_[i,1] )
                else:
                    fitnesses.append( fitnesses_[i,0] )

            ### DELETE
            self.pickle_logs['fitness_values'].append(all_fitnesses_flattened)
            self.pickle_logs['weights'].append(copy.copy(self.current_parameters.cpu().numpy()))
            #self.pickle_logs['perturbed_weights'].append([copy.copy(self.current_parameters.cpu().numpy() + self.sigma*self.noise.get(noise_id, len(self.current_parameters))) for noise_id in noise_ids])
            import pickle
            with open('es_logs.pickle', 'wb') as f:
                pickle.dump(self.pickle_logs, f)
            ###

            gradient = np.zeros(self.current_parameters.shape, dtype=np.float32)
            for j in range(len(noise_ids)):
                perturbation_j = self.noise.get(noise_ids[j], len(self.current_parameters))
                gradient += perturbation_j * fitnesses[j]

            gradient = gradient / len(noise_ids) / self.sigma
            if self.use_antithetic_sampling:
                # In antithetic sampling, the gradient is 1/(2*sigma) instead of 1/sigma.
                gradient /= 2.0
            #self.current_parameters -= self.learning_rate * gradient
            #if np.isnan(self.current_parameters).any() or np.isinf(self.current_parameters).any():
            #    print('ERROR: nans or infs detected after ES update.')

            nn.utils.vector_to_parameters(self.current_parameters, self.model.parameters())
            self._optimizer.update(gradient)

            self.current_parameters = nn.utils.parameters_to_vector(self.model.parameters()).cpu().detach()

        new_parameters = self._comm.bcast(self.current_parameters, root=0)
        nn.utils.vector_to_parameters(new_parameters.to(self.device), self.model.parameters())
        self.current_parameters = copy.deepcopy(new_parameters)

        if not self.is_master:
            iter_loss = 0 #np.mean(worker_loss)
        return iter_loss