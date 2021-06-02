import torch
import os
import argparse
import time
import math
import sys
from overrides import overrides

from allennlp.data import PyTorchDataLoader
from allennlp.nn import util as nn_util
from allennlp.common.tqdm import Tqdm

from src.scripts.base_script import BaseScript

class ExhaustiveScript(BaseScript):

    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--eval_only', action='store_true')

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/exhaustive/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/exhaustive/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def all_binary_masks(self, n, k):
        '''Generate all binary masks of length n and sparsity k'''
        assert n >= k
        if n == k:
            return [[1] * n]
        if k == 0:
            return [[0] * n]

        result = []
        for mask in self.all_binary_masks(n-1, k-1):
            result.append([1] + mask)
        for mask in self.all_binary_masks(n-1, k):
            result.append([0] + mask)
            
        return result

    @overrides
    def run(self):
        args = self.initialize()

        reader = self.init_task_reader(args)
        data, vocab = self.read_data(args, reader)
        data_loader = PyTorchDataLoader(data, batch_size=1, shuffle=False)
        task_model = self.init_task_model(args, vocab)
        task_model = self.load_task_model(args, task_model)

        best_mask_dict = {}

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
            for batch in generator_tqdm:
                batch = nn_util.move_to_device(batch, args.cuda_device)
                doc_length = len(batch['metadata'][0]['doc_tokens'])
                instance_id = batch['metadata'][0]['instance_id']
                best_mask_dict[instance_id] = {}

                for sparsity in self.sparsity_list:
                    k = math.ceil(doc_length * sparsity)
                    if not args.eval_only:
                        mask_list = mask_list = self.all_binary_masks(doc_length, k)
                    else:
                        mask_list = [mask for mask in best_masks[instance_id][sparsity_strs[no]].values() if mask is not None]                    
                    min_suff, max_comp = self.update_search_metrics(args, task_model, batch, sparsity, mask_list)
                    best_mask_dict[instance_id][sparsity] = {'suff': min_suff.best_doc_mask, 'comp': max_comp.best_doc_mask}

                result_dict, result_str = self.format_metrics()
                generator_tqdm.set_description(result_str, refresh=False)

            end = time.time()
            result_dict['time'] = end - start
            result_dict['args'] = str(sys.argv)
            print('-----Final Result-----')
            print('results: ', result_dict)
            print('-----------')

            prefix = args.exp_name
            self.write_json_to_file(result_dict, self.base_dir, 'final_result.json', prefix)
            self.write_json_to_file(best_mask_dict, self.base_dir, 'best_masks.json', prefix)
            self.write_metrics_to_csv(self.base_dir, 'per_point_metrics.csv', prefix)

if __name__=='__main__':
    script = ExhaustiveScript()
    script.run()