import torch
from overrides import overrides
import os
import time
import math
import sys
import numpy as np
import json

from allennlp.data import PyTorchDataLoader
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util

from src.scripts.base_script import BaseScript

class IntegratedGradientScript(BaseScript):
    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--steps', type=int)
        parser.add_argument('--top_k_selection', type=str, choices=['exact_k', 'up_to_k'], default='exact_k')
        parser.add_argument('--baseline', type=str, choices=['zero', 'mask'])
        parser.add_argument('--input_type', type=str, choices=['one_hot', 'embeddings'])
        parser.add_argument('--eval_only', action='store_true')

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/integrated_gradient/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/integrated_gradient/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def integrated_gradients(self,
            document,
            inp, 
            target_label_index,
            predictions_and_gradients,
            baseline,
            steps=50):
        """
        function courtesy of https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py
        with superficial alterations
        """          
        if baseline is None:
           baseline = 0*inp
        assert(baseline.shape == inp.shape)

        # Scale input and compute gradients.
        predictions, grads = [], []
        for i in range(0, steps+1):
            # scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(0, steps+1)]
            scaled_input = baseline + (float(i)/steps)*(inp-baseline)
            pred, grad = predictions_and_gradients(document, scaled_input, target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>
            predictions.append(pred.item())
            grads.append(grad)
        
        # reformat
        predictions = np.array(predictions)
        grads = torch.stack([grad.squeeze() for grad in grads])
        inp, baseline = inp.cpu(), baseline.cpu()

        # Use trapezoidal rule to approximate the integral.
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = torch.mean(grads, dim=0)
        integrated_gradients = (inp-baseline)*avg_grads  # shape: <inp.shape>
        return integrated_gradients, predictions

    @overrides
    def run(self):
        args = self.initialize()

        reader = self.init_task_reader(args)
        data, vocab = self.read_data(args, reader)
        data_loader = PyTorchDataLoader(data, batch_size=1, shuffle=False)
        task_model = self.init_task_model(args, vocab)
        task_model = self.load_task_model(args, task_model)

        saliency_dict = {}
        relative_error_sum = 0
        absolute_error_sum = 0
        count = 0

        task_model.eval()
        iterator = iter(data_loader)
        generator_tqdm = Tqdm.tqdm(iterator)

        saliency_path = os.path.join(self.base_dir, f'{args.exp_name}_saliency_scores.json')
        if args.eval_only:
            assert os.path.exists(saliency_path)
            saliency_scores = json.load(open(saliency_path))

        def predictions_and_gradients(document, scaled_inputs, target_label_index):
            # define gradient function for IG
            if args.input_type == 'embeddings':     
                output_dict = task_model._forward(document, inputs_embeds=scaled_inputs, label=target_label_index, metadata=metadata)
            elif args.input_type == 'one_hot':
                output_dict = task_model._forward(document, one_hot_input=scaled_inputs, label=target_label_index, metadata=metadata)
            pred = output_dict['predicted_prob']                        
            grad = torch.autograd.grad(pred, scaled_inputs)[0]
            pred = pred.detach().cpu()
            grad = grad.detach().cpu()
            return pred, grad

        start = time.time()
        for i, batch in enumerate(generator_tqdm):            
            batch = nn_util.move_to_device(batch, args.cuda_device)
            metadata = batch["metadata"]
            document = batch["document"]
            label = batch['label']
            doc_length = len(metadata[0]['doc_tokens'])

            # run IG
            if not args.eval_only:
                # print()
                # print(i)
                # os.system('nvidia-smi')
                if args.input_type == 'one_hot':
                    inp = task_model._one_hot_get_ids(document, metadata=metadata)
                    baseline = task_model._one_hot_get_ids(document, metadata=metadata, ratio=0, baseline='mask')
                elif args.input_type == 'embeddings':
                    _, inp = task_model._one_hot_get_embedding(document, metadata=metadata, ratio=1, baseline='mask', return_inputs_embeds=True)
                    _, baseline = task_model._one_hot_get_embedding(document, metadata=metadata, ratio=0, baseline='mask', return_inputs_embeds=True)            
                saliency, preds = self.integrated_gradients(document,
                        inp=inp,
                        target_label_index=label,
                        predictions_and_gradients=predictions_and_gradients,
                        baseline=baseline,
                        steps=args.steps)
                saliency = saliency.sum(dim=-1) # sum over vocab / embedding dim, depending on args.input_style

                # statistics
                diff = preds[-1] - preds[0]
                # print("input: ", torch.max(baseline, dim=-1))
                # print("input length: ", baseline.size(1))
                # print(preds)
                error = saliency.sum().item() - diff
                if diff > .02:
                    # print()
                    # print(f"Diff: {diff:.3f}")
                    # print(f"Saliency sum: {saliency.sum().item():.3f}")
                    absolute_error_sum += np.abs(error)
                    relative_error_sum += np.abs(error/diff)
                    # print(f"Running absolute error: {absolute_error_sum/(count+1):.4f}")
                    # print(f"Running relative error: {relative_error_sum/(count+1):.4f}")
                    count += 1
            
                # take document tokens (ie excluding query, if there is one)
                saliency = saliency[..., 1:(doc_length+1)]
                # Stack with 0 to become a 2d saliency score. Similar to Lime.
                saliency = saliency.squeeze(dim=0).unsqueeze(dim=-1)
                saliency = torch.cat([torch.zeros_like(saliency), saliency], dim=-1)

            else:
                saliency = torch.tensor(saliency_scores[metadata[0]['instance_id']])
        
            instance_id = metadata[0]['instance_id']
            saliency_dict[instance_id] = saliency.tolist()

            self.update_saliency_metrics(args, task_model, batch, saliency)
            result_dict, result_str = self.format_metrics()
            generator_tqdm.set_description(result_str, refresh=False)

        end = time.time()
        result_dict['time'] = end - start
        result_dict['args'] = str(sys.argv)
        print('-----Final Result-----')
        print('results: ', result_dict)
        print('-----------')
        if not args.eval_only:
            print(f"absolute error: {absolute_error_sum/(count+1):.4f}")
            print(f"relative error: {relative_error_sum/(count+1):.4f}")

        prefix = args.exp_name
        self.write_json_to_file(result_dict, self.base_dir, 'final_result.json', prefix)
        self.write_json_to_file(saliency_dict, self.base_dir, 'saliency_scores.json', prefix)
        self.write_metrics_to_csv(self.base_dir, 'per_point_metrics.csv', prefix)

if __name__=='__main__':
    script = IntegratedGradientScript()
    script.run()