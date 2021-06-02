import torch
from overrides import overrides
import os
import time
import math
import sys
import numpy as np
import random

from allennlp.data import PyTorchDataLoader
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util

from src.scripts.base_script import BaseScript

class VanillaGradientScript(BaseScript):

    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--mode', type=str, choices=['attention_mask', 'input_embeddings'])
        parser.add_argument('--top_k_selection', type=str, choices=['exact_k', 'up_to_k'])
        parser.add_argument('--eval_only', action='store_true')

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/vanilla_gradient/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/vanilla_gradient/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def gradient_saliency(self, args, task_model, batch, mode):
        assert mode in ['attention_mask', 'input_embeddings']
        assert batch['document']['task']['token_ids'].size(0) == 1
        doc_length = len(batch['metadata'][0]['doc_tokens'])
        if mode == 'attention_mask':
            attention_mask = batch['document']['task']['mask'].float()
            attention_mask.requires_grad_()
            batch['document']['task']['mask'] = attention_mask
            output_dict = task_model._forward(**batch)
            output_dict['predicted_prob'].backward()
            saliency = attention_mask.grad
        else:
            _, input_embeddings = task_model._one_hot_get_embedding(batch['document'], batch['metadata'], return_inputs_embeds=True)
            input_embeddings.requires_grad_()
            input_embeddings.retain_grad()
            output_dict = task_model._forward(
                document=batch['document'], always_keep_mask=batch['always_keep_mask'], label=batch['label'], 
                metadata=batch['metadata'], inputs_embeds=input_embeddings)
            output_dict['predicted_prob'].backward()
            saliency = torch.sum(input_embeddings.grad, dim=-1)

        # Turn gradient saliency into 2d logits by adding a column of 0s.
        saliency = saliency[:, 1:(doc_length+1)]
        saliency = saliency.squeeze(dim=0).unsqueeze(dim=1)
        saliency = torch.cat([torch.zeros_like(saliency), saliency], dim=1)
        assert saliency.size(0) == doc_length and saliency.size(1) == 2

        return saliency

    @overrides
    def run(self):
        args = self.initialize(enable_warnings=False)


        reader = self.init_task_reader(args)
        data, vocab = self.read_data(args, reader)
        data_loader = PyTorchDataLoader(data, batch_size=1, shuffle=False)
        task_model = self.init_task_model(args, vocab)
        task_model = self.load_task_model(args, task_model)

        if args.eval_only:
            args.initialization = 'vanilla_gradient'
            saliency_dict = self.read_saliency(args)
        else:
            saliency_dict = {}

        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        task_model.eval()
        iterator = iter(data_loader)
        generator_tqdm = Tqdm.tqdm(iterator)

        start = time.time()
        for batch in generator_tqdm:
            batch = nn_util.move_to_device(batch, args.cuda_device)

            # Get saliency
            instance_id = batch['metadata'][0]['instance_id']
            if args.eval_only:
                saliency = torch.tensor(saliency_dict[instance_id], device=args.cuda_device).float()
            else:
                saliency = self.gradient_saliency(args, task_model, batch, args.mode)
                saliency_dict[instance_id] = saliency.tolist()

            self.update_saliency_metrics(args, task_model, batch, saliency)
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
        self.write_json_to_file(saliency_dict, self.base_dir, 'saliency_scores.json', prefix)
        self.write_metrics_to_csv(self.base_dir, 'per_point_metrics.csv', prefix)


if __name__=='__main__':
    script = VanillaGradientScript()
    script.run()