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
import pandas as pd

from allennlp.data import PyTorchDataLoader
from allennlp.nn import util as nn_util
from allennlp.common.tqdm import Tqdm

from src.scripts.base_script import BaseScript

class OrderedScript(BaseScript):

    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--initialization', type=str, choices=['random', 'lime', 'masking_model', 'gradient_search', 'integrated_gradient'])
        parser.add_argument('--num_to_search', type=int)
        parser.add_argument('--additive', action='store_true')
        parser.add_argument('--saliency_exp_name', type=str)
        parser.add_argument('--eval_only', action='store_true')
        parser.add_argument('--save_all_metrics', action='store_true')
        parser.add_argument('--top_k_selection', type=str, default='exact_k', choices=['exact_k', 'up_to_k'])

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/ordered/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/ordered/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def flip_positions(self, mask:List[int], flipped_positions:List[int]):
        result = copy.deepcopy(mask)
        for pos in flipped_positions:
            result[pos] = 1 - result[pos]
        return result

    def ordered_masks(self, probs: List[float], mask: List[int], max_length: int=None, additive=False):
        '''
        given per-position probabilities, find the most probable masks in order
        - if additive=True, treat probs as scores that are additive rather than probs to be multiplied (e.g., log probs, LIME output, integrated gradient output, etc.)
        '''
        max_length = max_length if max_length is not None else float('inf')
        indexed_probs_0 = [(1-p, [i]) for i, p in enumerate(probs) if mask[i] == 0]
        indexed_probs_1 = [(1-p, [i]) for i, p in enumerate(probs) if mask[i] == 1]
        indexed_probs = []

        def f(x1, x2):
            return x1*x2 if not additive else x1+x2

        for prob_0, index_0 in indexed_probs_0:
            for prob_1, index_1 in indexed_probs_1:
                p = f(prob_0, prob_1)
                indices = sorted(index_0 + index_1)
                indexed_probs.append((p, indices))
        indexed_probs.sort(reverse=True)

        queue = []
        indices_set = set()
        for p, indices in indexed_probs:
            if len(queue) >= max_length:
                break
            heapq.heappush(queue, (-p, indices))
            indices_set.add(str(indices))
            for old_p, old_indices in queue:
                if len(queue) >= max_length:
                    break
                new_p = f(old_p, p)
                new_indices = sorted(old_indices + indices)
                if set(old_indices).isdisjoint(indices) and str(new_indices) not in indices_set:
                    heapq.heappush(queue, (new_p, new_indices))
                    indices_set.add(str(new_indices))

        results = []
        for i in range(min(max_length, len(queue))):
            flipped_positions = heapq.heappop(queue)[1]
            results.append(self.flip_positions(mask, flipped_positions))

        return results

    @overrides
    def run(self):
        args = self.initialize()

        reader = self.init_task_reader(args)
        data, vocab = self.read_data(args, reader)
        data_loader = PyTorchDataLoader(data, batch_size=1, shuffle=False)
        task_model = self.init_task_model(args, vocab)
        task_model = self.load_task_model(args, task_model)
        additive = args.additive

        if args.saliency_exp_name is not None:
            saliency_dict = self.read_saliency(args, saliency_exp_name=args.saliency_exp_name)

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
                if args.saliency_exp_name is not None:
                    saliency_scores = torch.tensor(saliency_dict[str(instance_id)], device=args.cuda_device)
                else:
                    saliency_scores = torch.normal(mean=0, std=1, size=(doc_length, 2))
                
                probs = torch.softmax(saliency_scores, dim=-1) if not additive else saliency_scores
                positive_probs = probs[..., 1]
                
                for no, sparsity in enumerate(self.sparsity_list):
                    k = math.ceil(doc_length * sparsity)
                    if args.top_k_selection == 'up_to_k':
                        num_positive_scores = torch.sum(positive_probs > .5).item() if not additive else torch.sum(positive_probs > 0).item()
                        k = min(k, num_positive_scores)
                    _, indices = torch.topk(positive_probs, k)

                    doc_mask = torch.zeros(doc_length, dtype=torch.long, device=args.cuda_device)
                    doc_mask[indices] = 1
                    mask_probs = probs[torch.arange(probs.size(0)), doc_mask]
                    # start = time.time()
                    # print("Starting search: ")
                    if not args.eval_only:
                        mask_list = self.ordered_masks(mask_probs.tolist(), doc_mask.tolist(), args.num_to_search, additive=additive)
                    else:
                        mask_list = [mask for mask in best_masks[instance_id][sparsity_strs[no]].values() if mask is not None]
                        assert(len(mask_list) == 2), "some saved masks are null"
                    # print(f"Seach time: {time.time() - start}, time per point: {(time.time() - start)/args.num_to_search}")

                    # start = time.time()
                    # print("Starting metric updates: ")
                    min_suff, max_comp = self.update_search_metrics(
                        args, task_model, batch, sparsity, mask_list, initial_mask=doc_mask)
                    all_suff_metrics_dict[sparsity][idx] = min_suff.all_metrics
                    all_suff_woe_metrics_dict[sparsity][idx] = min_suff.all_metrics_woe
                    all_comp_metrics_dict[sparsity][idx] = max_comp.all_metrics
                    all_comp_woe_metrics_dict[sparsity][idx] = max_comp.all_metrics_woe
                    all_suff_metrics_best_so_far_dict[sparsity][idx] = min_suff.all_metrics_best_so_far
                    all_suff_woe_metrics_best_so_far_dict[sparsity][idx] = min_suff.all_metrics_best_so_far_woe
                    all_comp_metrics_best_so_far_dict[sparsity][idx] = max_comp.all_metrics_best_so_far
                    all_comp_woe_metrics_best_so_far_dict[sparsity][idx] = max_comp.all_metrics_best_so_far_woe
                    best_mask_dict[instance_id][sparsity] = {'suff': min_suff.best_doc_mask, 'comp': max_comp.best_doc_mask}
                    # print(f"Metric update time: {time.time() - start}, time per point: {(time.time() - start)/args.num_to_search}")

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
                            all_df = df
                        else:
                            all_df = all_df.append(df, ignore_index=True)                
                _filename = prefix[:-3] + filename # cut off 'sdx' seed name from prefix
                save_path = os.path.join(self.base_dir, _filename)
                all_df.to_csv(save_path, index=False)

if __name__=='__main__':
    script = OrderedScript()
    script.run()