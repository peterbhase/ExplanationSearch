import torch
from overrides import overrides
from typing import List
import os
import time
import math
import sys
import copy
import numpy as np
import json

from allennlp.data import PyTorchDataLoader
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util

from src.scripts.base_script import BaseScript, BestMetric

class HotflipScript(BaseScript):
    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--initialization', type=str, choices=['random', 'lime', 'masking_model'])
        parser.add_argument('--beam_width', type=int)
        parser.add_argument('--num_steps', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--eval_only', action='store_true')
        # parser.add_argument('--flip_ratio', type=float, 
        #     help='Portion of the input to try to flip.' 
        #     'If flip_ratio=0.2 and doc_length=100, then hotflip will try to flip 20 tokens to find the best mask.'
        #     'Note that the same token can be flipped multiple times during search.')

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/hotflip/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/hotflip/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _hotflip_masks(
        self, 
        masks: List[List[int]], 
        grads: List[List[float]], 
        beam_width: int, 
        objective='suff'
    ):
        assert len(masks) == len(grads)

        # If objective is sufficiency, we want to:
        #   1) flip the 1 token with the largest gradient to 0 to decrease loss as much as possible
        #   2) flip the 0 token with the smallest gradient to 1 to increase loss as little as possible
        # If objective is comprehensiveness, we want the opposite of the process above
        beam = []
        for i in range(len(masks)):
            mask, grad = masks[i], grads[i]
            assert len(mask) == len(grad)
            sorted_grad_1 = sorted(
                [(value, index) for index, value in enumerate(grad) if mask[index] == 1], reverse=(objective == 'suff')
            )
            sorted_grad_0 = sorted(
                [(value, index) for index, value in enumerate(grad) if mask[index] == 0], reverse=(objective == 'comp')
            )
            assert sorted_grad_1 and sorted_grad_0

            best_list = []

            # If objective is sufficiency, we want high scores. If objective is comprehensiveness, we want low socres.
            worst_score_in_best_list = float('-inf') if objective == 'suff' else float('inf')
            for grad_1, index_1 in sorted_grad_1:
                for grad_0, index_0 in sorted_grad_0:
                    score = grad_1 - grad_0
                    if (objective == 'suff' and score < worst_score_in_best_list) or (objective == 'comp' and score > worst_score_in_best_list):
                        break
                    best_list.append((score, i, (index_1, index_0)))
                    if len(best_list) > beam_width:
                        best_list.sort(reverse=(objective == 'suff'))
                        best_list.pop()
                        worst_score_in_best_list = best_list[beam_width-1][0]

            beam.extend(best_list)

        beam.sort(reverse=(objective == 'suff'))

        hotflip_masks = []
        for _, mask_index, (index_1, index_0) in beam[:beam_width]:
            new_mask = copy.deepcopy(masks[mask_index])
            new_mask[index_1] = 0
            new_mask[index_0] = 1
            hotflip_masks.append(new_mask)

        return hotflip_masks

    @overrides
    def run(self):
        args = self.initialize()

        reader = self.init_task_reader(args)
        data, vocab = self.read_data(args, reader)
        data_loader = PyTorchDataLoader(data, batch_size=1, shuffle=False)
        task_model = self.init_task_model(args, vocab)
        task_model = self.load_task_model(args, task_model)
        num_forward = 0

        if args.initialization != 'random':
            saliency_dict = self.read_saliency(args)

        task_model.eval()
        iterator = iter(data_loader)
        generator_tqdm = Tqdm.tqdm(iterator)
        best_mask_dict = {}

        start = time.time()
        for num_point, batch in enumerate(generator_tqdm):            
            batch = nn_util.move_to_device(batch, args.cuda_device)
            batch_copy = copy.deepcopy(batch)
            metadata = batch['metadata'][0]
            document = batch['document']
            label = batch['label']

            instance_id = metadata['instance_id']
            doc_length = len(metadata['doc_tokens'])
            original_mask = document['task']['mask']
            best_mask_dict[instance_id] = {sparsity : {} for sparsity in self.sparsity_list}

            if args.initialization != 'random':
                saliency = torch.tensor(saliency_dict[str(instance_id)], device=args.cuda_device)
            else:
                saliency = torch.normal(mean=0, std=1, size=(doc_length, 2))
            probs = torch.softmax(saliency, dim=-1)
            positive_probs = probs[..., 1]

            with torch.no_grad():
                full_output_dict = task_model._forward(**batch)
            full_predicted_label = full_output_dict['predicted_label']

            for objective in ['suff', 'comp']:

                masks_path = os.path.join(self.base_dir, f'{args.exp_name}_best_masks.json')
                if args.eval_only:
                    assert os.path.exists(masks_path)
                    best_masks = json.load(open(masks_path))
                    sparsity_strs = ['0.05', '0.1', '0.2', '0.5']

                for sparsity in self.sparsity_list:
                    if objective == 'suff':
                        best_metric = BestMetric(objective='min')
                    else:
                        best_metric = BestMetric(objective='max')

                    k = math.ceil(sparsity * doc_length)
                    _, indices = torch.topk(positive_probs, k)

                    # If objective is sufficiency, we keep the most probable tokens
                    # If objective is comprehensiveness, we discard the most probable tokens
                    doc_mask = torch.zeros(doc_length, dtype=torch.long, device=args.cuda_device)
                    doc_mask[indices] = 1
                    if objective == 'comp':
                        doc_mask = 1 - doc_mask

                    doc_masks = [doc_mask.tolist()]

                    # num_steps = math.ceil(doc_length * args.flip_ratio)
                    num_steps = args.num_steps
                    for step in range(num_steps):
                        grads = []
                        for doc_mask in doc_masks:

                            # Hotflip forward pass
                            doc_mask = torch.tensor(doc_mask, device=args.cuda_device).float()
                            doc_mask.requires_grad_()
                            current_mask = original_mask.detach().clone().float()
                            current_mask[..., 1:(1+doc_length)] = doc_mask

                            document['task']['mask'] = current_mask
                            hotflip_output_dict = task_model._forward(document=document, label=full_predicted_label)
                            num_forward += 1
                            best_metric.from_output_dict(hotflip_output_dict, index=0, mask_list=[doc_mask.tolist()])
                            loss = hotflip_output_dict['loss']
                            doc_grad = torch.autograd.grad(loss, doc_mask)[0]
                            num_forward += 1

                            grads.append(doc_grad.tolist())

                        doc_masks = self._hotflip_masks(doc_masks, grads, args.beam_width, objective)
                    best_mask_dict[instance_id][sparsity].update({objective: best_metric.best_doc_mask})
                    self.metrics_dict[objective][sparsity].acc(best_metric.best_logits, batch['label'])
                    self.metrics_dict[objective][sparsity].metrics.append(
                        (full_output_dict['predicted_prob'] - best_metric.best_label_prob).item()
                    )
                    self.metrics_dict[objective][sparsity].woe_diffs.append(
                        (self.weight_of_evidence(full_output_dict['predicted_prob']) - self.weight_of_evidence(hotflip_output_dict['label_prob'])).item()
                    )
                    self.metrics_dict[objective][sparsity].pred_probs.append(best_metric.best_label_prob.item())

            # print(f"\nnum forward per point: {num_forward/(num_point+1)}")

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


if __name__ == '__main__':
    script = HotflipScript()
    script.run()