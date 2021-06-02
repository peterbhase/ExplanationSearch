import torch
import os
import argparse
import time
import math
import sys
from overrides import overrides
from typing import List, Tuple
import heapq
import copy
import json
import numpy as np
import pandas as pd

from allennlp.data import PyTorchDataLoader
from allennlp.nn import util as nn_util
from allennlp.common.tqdm import Tqdm

from src.scripts.base_script import BaseScript

class RandomScript(BaseScript):

    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--initialization', type=str, default='random')
        parser.add_argument('--num_to_search', type=int)
        parser.add_argument('--eval_only', action='store_true')
        parser.add_argument('--save_all_metrics', action='store_true')
        parser.add_argument('--search_space', type=str, choices=['exact_k', 'up_to_k'], help='max sparsity of mask space to search over')

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/random/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/random/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

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
        sample_size = math.ceil(sparsity*max_length)
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
        return list_of_masks

    @overrides
    def run(self):
        args = self.initialize()

        reader = self.init_task_reader(args)
        data, vocab = self.read_data(args, reader)
        data_loader = PyTorchDataLoader(data, batch_size=1, shuffle=False)
        task_model = self.init_task_model(args, vocab)
        task_model = self.load_task_model(args, task_model)

        best_mask_dict = {}
        all_suff_metrics_dict = {sparsity : {} for sparsity in self.sparsity_list}
        all_suff_woe_metrics_dict = {sparsity : {} for sparsity in self.sparsity_list}
        all_comp_metrics_dict = {sparsity : {} for sparsity in self.sparsity_list}
        all_comp_woe_metrics_dict = {sparsity : {} for sparsity in self.sparsity_list}
        all_suff_metrics_best_so_far_dict = {sparsity : {} for sparsity in self.sparsity_list}
        all_suff_woe_metrics_best_so_far_dict = {sparsity : {} for sparsity in self.sparsity_list}
        all_comp_metrics_best_so_far_dict = {sparsity : {} for sparsity in self.sparsity_list}
        all_comp_woe_metrics_best_so_far_dict = {sparsity : {} for sparsity in self.sparsity_list}
        

        masks_path = os.path.join(self.base_dir, f'{args.exp_name}_best_masks.json')
        if args.eval_only:
            assert os.path.exists(masks_path)
            best_masks = json.load(open(masks_path))
            sparsity_strs = ['0.05', '0.1', '0.2', '0.5']

        with torch.no_grad():
            task_model.eval()
            iterator = iter(data_loader)
            generator_tqdm = Tqdm.tqdm(iterator)

            start = time.time()            
            for idx, batch in enumerate(generator_tqdm):
                batch = nn_util.move_to_device(batch, args.cuda_device)
                metadata = batch['metadata'][0]

                doc_length = len(metadata['doc_tokens'])
                instance_id = metadata['instance_id']
                best_mask_dict[instance_id] = {}
 
                for no, sparsity in enumerate(self.sparsity_list):

                    # start = time.time()
                    # print("Starting search: ")
                    if not args.eval_only:
                        mask_list = self.random_masks(num_masks=args.num_to_search, max_length=doc_length, sparsity=sparsity, search_space=args.search_space)
                    else:
                        mask_list = [mask for mask in best_masks[instance_id][sparsity_strs[no]].values() if mask is not None]
                    # print(f"Seach time: {time.time() - start}, time per point: {(time.time() - start)/args.num_to_search}")

                    # print("Starting metric updates: ")
                    start = time.time()
                    min_suff, max_comp = self.update_search_metrics(
                        args, task_model, batch, sparsity, mask_list)
                    if args.save_all_metrics:
                        all_suff_metrics_dict[sparsity][idx] = min_suff.all_metrics
                        all_suff_woe_metrics_dict[sparsity][idx] = min_suff.all_metrics_woe
                        all_comp_metrics_dict[sparsity][idx] = max_comp.all_metrics
                        all_comp_woe_metrics_dict[sparsity][idx] = max_comp.all_metrics_woe
                        all_suff_metrics_best_so_far_dict[sparsity][idx] = min_suff.all_metrics_best_so_far
                        all_suff_woe_metrics_best_so_far_dict[sparsity][idx] = min_suff.all_metrics_best_so_far_woe
                        all_comp_metrics_best_so_far_dict[sparsity][idx] = max_comp.all_metrics_best_so_far
                        all_comp_woe_metrics_best_so_far_dict[sparsity][idx] = max_comp.all_metrics_best_so_far_woe
                    best_mask_dict[instance_id][sparsity] = {'suff': min_suff.best_doc_mask, 'comp': max_comp.best_doc_mask}
                    # print(f"Metric update time: {(time.time() - start):.2f}, time per point: {(time.time() - start)/args.num_to_search:.2f}")

                result_dict, result_str = self.format_metrics()
                generator_tqdm.set_description(result_str, refresh=False)

        end = time.time()
        result_dict['time'] = end - start
        result_dict['args'] = str(sys.argv)
        print('-----Final Result-----')
        for k, v in result_dict.items():
            print(f"  {k} : {v}")
        print('----------')

        prefix = args.exp_name
        self.write_json_to_file(result_dict, self.base_dir, 'final_result.json', prefix)
        self.write_json_to_file(best_mask_dict, self.base_dir, 'best_masks.json', prefix)
        self.write_metrics_to_csv(self.base_dir, 'per_point_metrics.csv', prefix)
        if args.save_all_metrics and not args.eval_only:
            # import pdb; pdb.set_trace()
            self.write_trajectories_to_csv(
                args,
                {'suff' : all_suff_metrics_dict,
                 'suff_woe' : all_suff_woe_metrics_dict,
                 'comp' : all_comp_metrics_dict,
                 'comp_woe' : all_comp_woe_metrics_dict
                }, 
                self.base_dir, 
                'all_samples.csv', 
                prefix
            )
            self.write_trajectories_to_csv(
                args,
                {'suff' : all_suff_metrics_best_so_far_dict,
                 'suff_woe' : all_suff_woe_metrics_best_so_far_dict,
                 'comp' : all_comp_metrics_best_so_far_dict,
                 'comp_woe' : all_comp_woe_metrics_best_so_far_dict
                }, 
                self.base_dir, 
                'all_trajectories.csv', 
                prefix
            )
            # combine all seeds into single df
            for filename in ['all_samples.csv', 'all_trajectories.csv']:
                all_df = None
                for seed in range(10):
                    exp_name = prefix[:-4] + f'_sd{str(seed)}'
                    _filename = exp_name + '_' + filename
                    filepath = os.path.join(self.base_dir, _filename)
                    if os.path.exists(filepath):
                        df = pd.read_csv(filepath)
                        df['seed'] = seed
                        if all_df is None:         
                            all_df = df.copy()
                        else:
                            all_df = all_df.append(df, ignore_index=True)                
                _filename = prefix[:-3] + filename # cut off 'sdx' seed name from prefix
                save_path = os.path.join(self.base_dir, _filename)
                all_df.to_csv(save_path, index=False)

            # save as json
            # self.write_json_to_file(all_suff_metrics_dict, self.base_dir, 'all_suff.json', prefix)
            # self.write_json_to_file(all_comp_metrics_dict, self.base_dir, 'all_comp.json', prefix)

if __name__=='__main__':
    script = RandomScript()
    script.run()
