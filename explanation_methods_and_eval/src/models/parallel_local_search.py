import numpy
import torch
import math
import numpy as np
import time
import operator as op
from functools import reduce

class PLSExplainer():
    def __init__(self, objective_function, target_sparsity, eval_budget, dimensionality, restarts=1, temp_decay = .8, search_space='up_to_k', no_duplicates=True):
        '''
        objective_function : maps from np.ndarray of shape 1 x p -> (suff/comp, suff_woe/comp_woe)
        search_space: up_to_k means max sparsity is ceil(dim * target_sparsity). if set to exact_k, then will be exactly k-sparse
        no_duplicates: if True, never evaluate an x twice. requires keeping track of all previous x
        '''
        self.objective_function = objective_function
        self.target_sparsity = target_sparsity
        self.eval_budget = eval_budget
        self.dimensionality = dimensionality
        self.search_space = search_space
        self.max_sparsity = math.ceil(self.dimensionality*self.target_sparsity)
        assert target_sparsity > 0 and target_sparsity < 1

        # limit both restarts and per_restart budget based on the number of possible explanations
        self.num_possible_explanations = self.ncr(self.dimensionality, self.max_sparsity)
        self.restarts = min(restarts, self.num_possible_explanations)
        self.n_iters = math.ceil(self.eval_budget / self.restarts)
        self.n_iters = min(self.n_iters, math.ceil(self.num_possible_explanations / self.restarts)) # limit n_iters per restart to not add up to more than num_possible_explanations        

        self.random_masks = self.random_masks(num_masks=eval_budget, max_length=dimensionality, sparsity=target_sparsity, search_space='exact_k')
        self.remaining_sample_idx = list(range(len(self.random_masks)))
        self.no_duplicates = no_duplicates

        self.T = 1.
        self.temp_decay = temp_decay
        self.seen_masks = set()

    def balanced_array(self, size, prop):
        # make array of 0s and 1s of len=size and mean ~= prop
        array = np.zeros(size)
        where_ones = np.random.choice(np.arange(size), size=math.ceil(size*prop), replace=False)
        array[where_ones] = 1
        return array

    def ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom  # or / in Python 2

    def run(self):

        (all_masks, all_obj, all_woe_obj) = self.parallel_local_search()

        return_masks = all_masks[-1].reshape(1, -1) # single best mask
        return_obj = all_obj
        return_woe_obj = all_woe_obj
        self.objective_at_t = self.create_objective_over_time(return_obj)
        self.objective_at_t_woe = self.create_objective_over_time(return_woe_obj)
        return (return_masks, return_obj, return_woe_obj)

    def create_objective_over_time(self, obj):
        # takes obj, array of length n, and repeats each element in obj self.restarts time
        # e.g., [0,1,2], with self.restarts=2, becomes [0,0,1,1,2,2]
        return obj.repeat(self.restarts, axis=0)

    def sample_single_x(self, old_x=None, add_to_seen=True):
        # sample new_x from old_x, nparray of length n_vars, or from self.random_masks
        # first of all, if no new explanations left to sample, generate random one        
        if len(self.seen_masks) >= self.num_possible_explanations:
            new_x = self.balanced_array(size=self.dimensionality, prop=self.target_sparsity)
        elif old_x is None:
            sample_idx = np.random.choice(self.remaining_sample_idx)
            if self.no_duplicates:
                self.remaining_sample_idx.remove(sample_idx)
            new_x = self.random_masks[sample_idx]
        else:
            if self.search_space == 'up_to_k':
                flip_bit = np.random.randint(self.dimensionality)
                new_x = old_x.copy()
                new_x[flip_bit] = 1. - new_x[flip_bit]
                exceeding_max_sparsity = (np.sum(new_x) > self.max_sparsity)
                if exceeding_max_sparsity:
                    where_one = np.argwhere(new_x == 1).reshape(-1)
                    where_one = np.setdiff1d(where_one, flip_bit)
                    flip_other_bit = np.random.choice(where_one, size=1)
                    new_x[flip_other_bit] = 0
            if self.search_space == 'exact_k':
                where_ones_orig = np.argwhere(old_x==1).reshape(-1)
                where_zeros_orig = np.argwhere(old_x==0).reshape(-1)
                flip_bit = np.random.choice(where_zeros_orig, size=1)
                new_x = old_x.copy()
                new_x[flip_bit] = 1
                flip_back = np.random.choice(where_ones_orig, size=1)
                new_x[flip_back] = 0
        if self.no_duplicates and add_to_seen:
            self.seen_masks.add(str(new_x))
        return new_x

    def sample_new_x(self, old_x):
        '''
        old_x : np.array of shape n x n_vars
        - samples one new x per array in old_x that is not seen yet in self.seen_masks (up to 1k resamples, then just move on)
        - new samples added to seen_masks as we go
        '''
        start = time.time()
        resample_counter = 0
        eligible_positions = np.arange(self.dimensionality)
        new_x = old_x.copy()
        for i in range(self.restarts):
            if self.no_duplicates:
                proposal = self.sample_single_x(new_x[i], add_to_seen=False)
                while str(proposal) in self.seen_masks:
                    proposal = self.sample_single_x(proposal, add_to_seen=False) # random walk resampling
                    resample_counter += 1
                    if resample_counter > 100:
                        resample_counter = 0
                        break
                new_x[i] = proposal
                self.seen_masks.add(str(new_x[i])) 
                if len(self.seen_masks) == self.num_possible_explanations:
                   break
            else:
                new_x[i] = self.sample_single_x(new_x[i], add_to_seen=False)
        # print(f"{(time.time()-start):.2f} seconds for sample")
        return new_x

    def parallel_local_search(self):
        '''
        runs self.restarts runs of local_search search for self.eval_budget // self.num_restarts steps
        track both suff/comp obj and WoE version
        '''

        # Extract inputs
        n_vars = self.dimensionality
        n_iter = self.n_iters
        T = self.T
        start = time.time()

        # Declare vectors to save solutions
        model_iter   = np.zeros((n_iter,n_vars))
        obj_iter     = np.zeros(n_iter)
        obj_woe_iter = np.zeros(n_iter)

        # Set initial condition and evaluate objective
        old_x   = np.array([self.sample_single_x() for i in range(self.restarts)])        
        old_obj, old_obj_woe = self.objective_function(old_x)

        # Set best_x and best_obj
        best_idx = np.argmin(old_obj)
        best_x   = old_x[best_idx]
        best_obj = old_obj[best_idx]
        best_obj_woe = old_obj_woe[best_idx]

        # set first iter location
        model_iter[0,:] = best_x
        obj_iter[0]     = best_obj
        obj_woe_iter[0] = best_obj_woe

        # Run simulated annealing
        for t in range(1,n_iter):   

            # Decrease T according to cooling schedule
            T *= self.temp_decay

            # Find new samples
            new_x = self.sample_new_x(old_x)

            # Evaluate objective function
            new_obj, new_obj_woe = self.objective_function(new_x)

            # Update current solution iterate
            better_than = (new_obj < old_obj)            
            if np.any(better_than):
                where_better = np.argwhere(better_than).reshape(-1)
                for pick_idx in where_better:
                    old_x[pick_idx]   = new_x[pick_idx]
                    old_obj[pick_idx] = new_obj[pick_idx]
                    old_obj_woe[pick_idx] = new_obj_woe[pick_idx]
            where_not_better = np.argwhere(1 - better_than).reshape(-1)
            for idx in where_not_better:
                auto_acceptance_prob = np.exp((old_obj[idx] - new_obj[idx])/T)
                if (np.random.rand() < auto_acceptance_prob):
                    old_x[idx] = new_x[idx]
                    old_obj[idx] = new_obj[idx]
                    old_obj_woe[idx] = new_obj_woe[idx]
            # set best_x
            best_idx = np.argmin(new_obj)
            if new_obj[best_idx] < best_obj:
                best_x   = new_x[best_idx]
                best_obj = new_obj[best_idx]
                best_obj_woe = new_obj_woe[best_idx]

            # save solution
            model_iter[t,:] = best_x
            obj_iter[t]     = best_obj
            obj_woe_iter[t] = best_obj_woe

            if len(self.seen_masks) == self.num_possible_explanations:
                break

        # print(f"{(time.time()-start):.2f} seconds for SA with {n_iter} steps")
        # remove any all-zero x, which are un-imputed values caused by sampling ending early when the space space size is hit
        where_non_zero = np.argwhere(np.sum(model_iter, axis=1) > 0).reshape(-1)
        model_iter = model_iter[where_non_zero, :]
        obj_iter = obj_iter[where_non_zero]
        obj_woe_iter = obj_woe_iter[where_non_zero]
        
        return (model_iter, obj_iter, obj_woe_iter)

    def local_search(self):
        '''
        runs full local_search search for self.eval_budget steps
        track both suff/comp obj and WoE version
        '''

        # Extract inputs        
        n_vars = self.dimensionality
        T = self.T
        resample_counter = 0

        # Declare vectors to save solutions
        model_iter   = np.zeros((self.n_iters, n_vars))
        obj_iter     = np.zeros(self.n_iters)
        obj_woe_iter = np.zeros(self.n_iters)

        # Set initial condition and evaluate objective
        old_x = self.sample_single_x()
        old_obj, old_obj_woe = self.objective_function(old_x)
        self.seen_masks.add(str(old_x))

        # Set best_x and best_obj
        best_x   = old_x
        best_obj = old_obj
        best_obj_woe = old_obj_woe

        # set first iter location
        model_iter[0,:] = best_x
        obj_iter[0]     = best_obj
        obj_woe_iter[0] = best_obj_woe

        # Run simulated annealing
        for t in range(1, self.n_iters):

            # Decrease T according to cooling schedule
            T *= self.temp_decay            

            # Find new sample -- with ceiling on k-sparsity
            new_x = self.sample_single_x(old_x)
            while str(new_x) in self.seen_masks:
                new_x = self.sample_single_x(old_x)
                resample_counter += 1
                if resample_counter > 100:
                    resample_counter = 0
                    break
            self.seen_masks.add(str(new_x))

            # Evaluate objective function
            start = time.time()
            new_obj, new_obj_woe = self.objective_function(new_x)

            # Update current solution iterate
            auto_acceptance_prob = np.exp((old_obj - new_obj)/T)
            if (new_obj < old_obj) or (np.random.rand() < auto_acceptance_prob):
                old_x   = new_x
                old_obj = new_obj
                old_obj_woe = new_obj_woe

            # Update best solution
            if new_obj < best_obj:
                best_x   = new_x
                best_obj = new_obj
                best_obj_woe = new_obj_woe

            # save solution
            model_iter[t,:] = best_x
            obj_iter[t]     = best_obj
            obj_woe_iter[t] = best_obj_woe

            # print(f"{(time.time()-start):.2f} seconds for step {t}")
            if len(self.seen_masks) == self.num_possible_explanations:
                break

        return (model_iter, obj_iter, obj_woe_iter)


    def random_masks(self, num_masks: int, max_length: int, sparsity: float, search_space : str):

        def binomial(n, k):
            if not 0 <= k <= n:
                return 0
            b = 1
            for t in range(min(k, n-k)):
                b *= n
                b //= t+1
                n -= 1
            return b

        list_of_masks = []
        sample_size = math.ceil(sparsity*max_length) # not n, but length of explanation = sum(mask)
        space_size = binomial(max_length, sample_size)
        
        if space_size < 100000:            
            list_of_masks = self.all_binary_masks(max_length, sample_size, search_space)
            np.random.shuffle(list_of_masks)
            list_of_masks = list_of_masks[:num_masks]

        # sample until num_masks sampled or space_size is hit
        else:
            failed_proposals = 0
            while len(list_of_masks) < num_masks:
                mask = np.zeros(max_length)
                size = sample_size if search_space=='exact_k' else np.random.randint(1, sample_size+1)
                where_one = np.random.choice(np.arange(1,max_length), size=size, replace=False) # never mask first position, corresponding to special token
                mask[where_one] = 1
                mask = mask.tolist()
                if mask not in list_of_masks:
                    list_of_masks.append(mask)
                else:
                    failed_proposals += 1
                if failed_proposals > 20000:
                    break
        return np.array(list_of_masks)


    def all_binary_masks(self, n, k, search_space='exact_k'):
        '''Generate all binary masks of length n and sparsity k if search_space=exact_k, or up to sparsity k if search_space=up_to_k'''
        assert n >= k
        if n == k:
            return [[1] * n]
        if k == 0:
            return [[0] * n]

        result = []
        for mask in self.all_binary_masks(n-1, k-1):
            result.append([1] + mask)
            if search_space=='up_to_k':                
                result.append([0] + mask)
        for mask in self.all_binary_masks(n-1, k):
            result.append([0] + mask)            
            
        return result