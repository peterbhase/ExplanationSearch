import torch
from overrides import overrides
import os
import time
import math
import sys
import numpy as np
import random
from lime.lime_text import LimeTextExplainer

from allennlp.data import PyTorchDataLoader
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util

from src.scripts.base_script import BaseScript

class LimeScript(BaseScript):

    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--threshold', type=float)
        parser.add_argument('--num_samples', type=int)
        parser.add_argument('--top_k_selection', type=str, choices=['exact_k', 'up_to_k'])
        parser.add_argument('--eval_only', action='store_true')

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/lime/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/lime/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def lime_saliency(self, args, task_model, batch):
        output_labels = task_model.vocab.get_token_to_index_vocabulary('labels').keys()
        lime_explainer = LimeTextExplainer(class_names=output_labels, split_expression=' ', bow=False)

        with torch.no_grad():
            full_output_dict = task_model._forward(**batch)

        batch_document, batch_predicted_label = \
            self.stack_input(batch['document'], full_output_dict['predicted_label'], args.batch_size)
        doc_tokens = batch['metadata'][0]['doc_tokens']
        doc_length = len(doc_tokens)

        def predict_proba(text_list):
            binary_masks = [[1 if t != 'UNKWORDZ' else 0 for t in tokens.split(' ')] for tokens in text_list]

            probs = []
            for i in range(0, len(binary_masks), args.batch_size):
                true_batch_size = min(len(binary_masks) - i, args.batch_size)
                for j in range(true_batch_size):
                    batch_document['task']['mask'][j, ..., 1:(doc_length+1)] = torch.tensor(binary_masks[i+j])
                with torch.no_grad():
                    output_dict = task_model._forward(document=batch_document, label=batch_predicted_label)
                probs.extend(output_dict['probs'][:true_batch_size].cpu().data.numpy())

            assert len(probs) == len(text_list)
            return np.vstack(probs)

        lime_text = ' '.join([str(i) for i, x in enumerate(doc_tokens)])
        num_features = math.ceil(args.threshold * doc_length)
        full_predicted_label = full_output_dict['predicted_label'].item()
        explanation = lime_explainer.explain_instance(
            lime_text,
            predict_proba,
            num_features=num_features,
            labels=(full_predicted_label,),
            num_samples=args.num_samples,
        )

        weights = explanation.as_list(full_predicted_label)
        saliency = [0.0 for i in range(doc_length)]
        for f, w in weights:
            saliency[int(f)] = w

        # Turn Lime saliency into 2d logits by adding a column of 0s
        saliency = torch.tensor(saliency).unsqueeze(dim=1)
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
            args.initialization = 'lime'
            saliency_dict = self.read_saliency(args)
        else:
            saliency_dict = {}

        # fix the lime seed! do not let vary with args.seed
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        with torch.no_grad():
            task_model.eval()
            iterator = iter(data_loader)
            generator_tqdm = Tqdm.tqdm(iterator)

            start = time.time()
            for batch in generator_tqdm:
                batch = nn_util.move_to_device(batch, args.cuda_device)

                # Get Lime saliency
                instance_id = batch['metadata'][0]['instance_id']
                if args.eval_only:
                    saliency = torch.tensor(saliency_dict[instance_id], device=args.cuda_device).float()
                else:
                    saliency = self.lime_saliency(args, task_model, batch)
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
    script = LimeScript()
    script.run()