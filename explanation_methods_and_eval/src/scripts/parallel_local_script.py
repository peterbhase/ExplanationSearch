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
import time

from allennlp.data import PyTorchDataLoader
from allennlp.nn import util as nn_util
from allennlp.common.tqdm import Tqdm

from src.scripts.base_script import BaseScript
from src.models.parallel_local_search import PLSExplainer

class PLSScript(BaseScript):

    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--method', type=str, default='SA', choices=['SA','BOCS'], help='SA = simulated annealing, BOCS is from https://arxiv.org/pdf/1806.08838.pdf')
        parser.add_argument('--num_to_search', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--num_restarts', type=int, help="times to rerun search from different starting points")
        parser.add_argument('--temp_decay', type=float, help="temp decay in SA")
        parser.add_argument('--eval_only', action='store_true')
        parser.add_argument('--save_all_metrics', action='store_true')
        parser.add_argument('--search_space', type=str, choices=['exact_k', 'up_to_k'], help='max sparsity of mask space to search over')


    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/PLS/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/PLS/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

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

            start_time = time.time()            
            for idx, batch in enumerate(generator_tqdm):
                batch = nn_util.move_to_device(batch, args.cuda_device)
                metadata = batch['metadata'][0]
                
                doc_length = len(metadata['doc_tokens'])
                instance_id = metadata['instance_id']
                best_mask_dict[instance_id] = {}

                full_output_dict = task_model._forward(**batch)
                orig_pred_prob = full_output_dict['predicted_prob'].item()
                orig_pred_label = full_output_dict['predicted_label']

                for objective in ['suff', 'comp']:
                    
                    # define objective function here to give to search_class below
                    def objective_function(mask):
                        '''
                        computes suff or comp on the current data point
                        mask : np.ndarray of shape (1,p) where p is number of tokens 
                        returns scalar value, the suff or comp of task_model computed on document using attention_mask=mask
                        ''' 
                        # flip mask if comp since sparsity always <= .5 (will be flipped in base_script)
                        if objective == 'comp':
                            mask = 1-mask

                        # split handles batching if too many parallel SA to fit into memory
                        num_batches = max(1,math.ceil(args.num_restarts / args.batch_size))
                        masks = np.array_split(mask, indices_or_sections=num_batches)
                        obj_vals, obj_val_woes = [], []
                        for mask in masks:
                            stacked_input, stacked_label = self.stack_input(document=batch['document'], label=orig_pred_label, batch_size=mask.shape[0])
                            doc_mask_tensor = torch.tensor(mask)
                            stacked_input["task"]["mask"][:, 1:(doc_length+1)] = doc_mask_tensor
                            start = time.time()
                            output_dict = task_model._forward(document=stacked_input, label=stacked_label)
                            # print(f"SA {(time.time()-start):.4f} forward time ")
                            pred_prob = output_dict['label_prob'].cpu().numpy()
                            if objective == 'suff':
                                obj_val = orig_pred_prob - pred_prob
                                obj_val_woe = self.np_woe(orig_pred_prob) - self.np_woe(pred_prob)
                            if objective == 'comp':
                                obj_val = -(orig_pred_prob - pred_prob)
                                obj_val_woe = -(self.np_woe(orig_pred_prob) - self.np_woe(pred_prob))
                            obj_vals.append(obj_val)
                            obj_val_woes.append(obj_val_woe)
                        obj_val = np.concatenate(obj_vals).reshape(-1)
                        obj_val_woe = np.concatenate(obj_val_woes).reshape(-1)
                        return obj_val, obj_val_woe

                    for no, sparsity in enumerate(self.sparsity_list):
                        num_possible_masks = self.ncr(doc_length, math.ceil(doc_length*sparsity))

                        if args.eval_only:
                            mask_list = [best_masks[instance_id][sparsity][objective]]
                        if not args.eval_only:

                            # define search method
                            search_class = PLSExplainer(
                                objective_function=objective_function, 
                                target_sparsity=sparsity, 
                                eval_budget=args.num_to_search,
                                dimensionality=doc_length,
                                restarts=args.num_restarts, # num parallel runs
                                temp_decay=args.temp_decay,
                                search_space=args.search_space,
                                no_duplicates=True)                        

                            start = time.time()
                            masks, obj_values, obj_woe_values = search_class.run()
                            # print(f"Search time: {(time.time() - start):.2f}, time per point: {(time.time() - start)/args.num_to_search:.2f}")
                            mask_list = masks.tolist()
                            best_mask = [mask_list[-1]]
                            if args.search_space == 'up_to_k': assert (sum(best_mask[0]) <= math.ceil(doc_length*sparsity))
                            if args.search_space == 'exact_k': assert (sum(best_mask[0]) == math.ceil(doc_length*sparsity))

                            start = time.time()
                            min_suff, max_comp = self.update_search_metrics(
                                args, task_model, batch, sparsity, best_mask, objective=objective)
                            if args.method=='SA':
                                if objective == 'suff':
                                    all_suff_metrics_best_so_far_dict[sparsity][idx] = search_class.objective_at_t.tolist()
                                    all_suff_woe_metrics_best_so_far_dict[sparsity][idx] = search_class.objective_at_t_woe.tolist()
                                    best_mask_dict[instance_id][sparsity] = {'suff': min_suff.best_doc_mask}
                                if objective == 'comp':
                                    all_comp_metrics_best_so_far_dict[sparsity][idx] = (-search_class.objective_at_t).tolist()
                                    all_comp_woe_metrics_best_so_far_dict[sparsity][idx] = (-search_class.objective_at_t_woe).tolist()
                                    best_mask_dict[instance_id][sparsity].update({'comp': max_comp.best_doc_mask})

                result_dict, result_str = self.format_metrics()
                generator_tqdm.set_description(result_str, refresh=False)

        end = time.time()
        result_dict['time'] = end - start_time
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
            for filename in ['all_trajectories.csv']:
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
    script = PLSScript()
    script.run()
