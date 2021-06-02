import torch
import sys
import os
import argparse
import time
import shutil
import json
from overrides import overrides

from allennlp.predictors import Predictor
from allennlp.data import PyTorchDataLoader
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.training.optimizers import AdamOptimizer, SgdOptimizer
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import (
    StepLearningRateScheduler, 
    ExponentialLearningRateScheduler, 
    LinearWithWarmup,
    CosineWithRestarts,
)
from allennlp.nn import util as nn_util
from allennlp.common.tqdm import Tqdm
from allennlp.training.trainer import TrackEpochCallback

from src.scripts.base_script import BaseScript
from src.models.gradient_search_model import GradientSearchModel

from src.custom_checkpointer import GradientSearchCheckpointer
# from allennlp.training.checkpointer import Checkpointer as GradientSearchCheckpointer

class GradientScript(BaseScript):
    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--initialization', type=str, 
            choices=['random', 'leave_one_out', 'lime', 'gradient_search', 'masking_model'], default='random')
        parser.add_argument('--top_k_selection', type=str, choices=['exact_k', 'up_to_k'], default='exact_k')
        parser.add_argument('--objective', type=str, choices=['suff', 'comp', 'both'])
        parser.add_argument('--num_samples', type=int)
        parser.add_argument('--steps_per_epoch', type=int)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--num_epochs', type=int)
        parser.add_argument('--patience', type=int)
        parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW'], default='AdamW')
        parser.add_argument('--scheduler', type=str, choices=['linear', 'cosine', 'step'])
        parser.add_argument('--task_model_dropout', action='store_true', default=False)
        parser.add_argument('--sparsity_weight', type=float)
        parser.add_argument('--sparsity', type=float)
        parser.add_argument('--skip_to', type=int)
        parser.add_argument('--eval_only', action='store_true')

    @overrides
    def check_args(self, args):
        assert (args.num_samples < 10) or (args.num_samples % 10 == 0)

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/gradient_search/{args.dataset}/{args.datasplit}'
        else:
            self.base_dir = 'outputs/gradient_search/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def init_optimizer(self, args, gradient_model):
        parameters = [
            [n, p]
            for n, p in gradient_model.named_parameters() if p.requires_grad
        ]
        parameter_groups = [(['task_model'], {'requires_grad': False})]
        if args.optimizer == 'SGD':
            optimizer = SgdOptimizer(model_parameters=parameters, parameter_groups=parameter_groups, lr=args.lr)
        elif args.optimizer == 'AdamW':
            optimizer = AdamOptimizer(model_parameters=parameters, parameter_groups=parameter_groups, lr=args.lr)
        else:
            raise ValueError('Optimizer not recognized')

        return optimizer

    def init_lr_scheduler(self, args, optimizer):
        if args.scheduler == 'linear':
            return LinearWithWarmup(optimizer, num_epochs=args.num_epochs, num_steps_per_epoch=args.steps_per_epoch, warmup_steps=0)
        elif args.scheduler == 'cosine':
            return CosineWithRestarts(optimizer, t_initial=10, t_mul=2)
        elif args.scheduler == 'step':
            return StepLearningRateScheduler(optimizer, step_size=100, gamma=0.5)
        elif args.scheduler is None:
            return None
        else:
            raise ValueError('Scheduler not recognized')

    def leave_one_out_saliency(self, args, task_model, data):
        '''Leave one out saliency initialization for gradient search.'''
        directory = os.path.join('outputs', 'leave_one_out', args.dataset, args.datasplit)
        filepath = os.path.join(directory, 'saliency_scores.json')
        if os.path.exists(filepath):
            with open(filepath) as input_file:
                initial_dict = json.load(input_file)
            if len(initial_dict) >= len(data):
                return initial_dict

        data_loader = PyTorchDataLoader(data, batch_size=1, shuffle=False)
        initial_dict = {}
        with torch.no_grad():
            generator_tqdm = Tqdm.tqdm(iter(data_loader))
            for batch in generator_tqdm:
                batch = nn_util.move_to_device(batch, args.cuda_device)
                document = batch['document']
                label = batch['label']
                metadata = batch['metadata'][0]
                doc_length = len(metadata['doc_tokens'])
                instance_id = metadata['instance_id']
                original_mask = document['task']['mask']

                full_output_dict = task_model._forward(document=document, label=label)

                saliency_scores = []
                for i in range(doc_length):
                    current_mask = original_mask.detach().clone()
                    current_mask[0, 1+i] = 0
                    document['task']['mask'] = current_mask
                    masked_output_dict = task_model._forward(document=document, label=label)
                    saliency_scores.append((full_output_dict['logits'] - masked_output_dict['logits']).squeeze(dim=0).tolist())

                initial_dict[instance_id] = saliency_scores

        self.write_json_to_file(initial_dict, directory, 'saliency_scores.json')
        return initial_dict

    @overrides
    def run(self):
        args = self.initialize()

        reader = self.init_task_reader(args)
        data, vocab = self.read_data(args, reader)
        task_model = self.init_task_model(args, vocab)
        task_model = self.load_task_model(args, task_model)

        initial_dict = None
        if args.initialization in ['leave_one_out']:
            initial_dict = self.leave_one_out_saliency(args, task_model, data)
        elif args.initialization in ['lime', 'gradient_search', 'masking_model']:
            initial_dict = self.read_saliency(args)

        task_predictor = Predictor(model=task_model, dataset_reader=reader)

        saliency_dict = {}
        epoch_metric_matrix = []

        start = time.time()
        for idx, instance in enumerate(data):
            if args.skip_to and idx < args.skip_to:
                continue
            train_batch_size = min(args.num_samples, 10)
            gradient_accumulation = args.num_samples // train_batch_size
            train_data_size = args.num_samples * args.steps_per_epoch

            train_data = AllennlpDataset([instance for i in range(train_data_size)], vocab=vocab)
            train_data_loader = PyTorchDataLoader(train_data, batch_size=train_batch_size, shuffle=False)
            validation_data = AllennlpDataset([instance], vocab=vocab)
            validation_data_loader = PyTorchDataLoader(validation_data, batch_size=1, shuffle=False)

            instance_id = instance.fields['metadata']['instance_id']
            doc_tokens = instance.fields['metadata']['doc_tokens']

            full_output_dict = task_predictor.predict_instance(instance)

            if initial_dict is None:
                initial_logits = torch.normal(mean=0, std=1, size=(len(doc_tokens), 2))
            else:
                initial_logits = torch.tensor(initial_dict[str(instance_id)])
            
            gradient_model = GradientSearchModel(
                cuda_device=args.cuda_device,
                vocab=vocab, 
                task_model=task_model,
                full_predicted_label=full_output_dict['predicted_label'],
                full_predicted_prob=full_output_dict['predicted_prob'],
                initial_logits=initial_logits,
                sparsity=args.sparsity,
                sparsity_weight=args.sparsity_weight,
                objective=args.objective,
                top_k_selection=args.top_k_selection,
                task_model_dropout=args.task_model_dropout
            )
            gradient_model.to(args.cuda_device)
            
            if not args.eval_only:
                optimizer = self.init_optimizer(args, gradient_model)
                lr_scheduler = self.init_lr_scheduler(args, optimizer)

                tmp_serialization_dir = f'outputs/gradient_search/tmp{args.cuda_device}'
                if os.path.exists(tmp_serialization_dir):
                    shutil.rmtree(tmp_serialization_dir)
                os.makedirs(tmp_serialization_dir, exist_ok=True)
                checkpointer = GradientSearchCheckpointer(serialization_dir=tmp_serialization_dir, num_serialized_models_to_keep=1)
                callback = TrackEpochCallback()

                trainer = GradientDescentTrainer(
                    model=gradient_model, 
                    data_loader=train_data_loader, 
                    checkpointer=checkpointer,
                    validation_data_loader=validation_data_loader, 
                    learning_rate_scheduler=lr_scheduler,
                    validation_metric='-loss',
                    num_epochs=args.num_epochs, 
                    patience=args.patience,
                    optimizer=optimizer, 
                    cuda_device=args.cuda_device,
                    grad_norm=10.0,
                    epoch_callbacks=None,#[callback],
                    num_gradient_accumulation_steps=gradient_accumulation
                )
                trainer.train()

                epoch_scores = [gradient_model.get_epoch_metric_dict()[i].item() for i in range(len(gradient_model.get_epoch_metric_dict()))]
                while len(epoch_scores) < args.num_epochs: epoch_scores += ['NA']
                epoch_metric_matrix.append(epoch_scores)
                gradient_model.load_state_dict(torch.load(os.path.join(tmp_serialization_dir, 'best.th')), strict=False)

            gradient_predictor = Predictor(model=gradient_model, dataset_reader=reader)
            gradient_output_dict = gradient_predictor.predict_instance(instance)
            saliency = gradient_output_dict['saliency']
            saliency_dict[instance.fields['metadata']['instance_id']] = saliency

            for batch in iter(validation_data_loader):
                batch = nn_util.move_to_device(batch, args.cuda_device)
                self.update_saliency_metrics(args, task_model, batch, torch.tensor(saliency))

            result_dict, result_str = self.format_metrics()
            print('----------')
            print('metrics: ', result_str)
            print('max_memory_usage: {:.0f} MB'.format(torch.cuda.max_memory_allocated(device=args.cuda_device) / (1024**2)))
            print('progress: {:.3f}'.format((idx + 1) / len(data)))
            print('----------')

            task_model.zero_grad()
            torch.cuda.empty_cache()

        end = time.time()
        result_dict['time'] = end - start
        result_dict['args'] = str(sys.argv)
        print('-----Final Result-----')
        for k, v in result_dict.items():
            print(f"  {k} : {v}")
        print('----------')

        prefix = args.exp_name        
        self.write_json_to_file(result_dict, self.base_dir, 'final_result.json', prefix)
        self.write_fidelity_vector(self.base_dir, prefix=prefix)
        if not args.eval_only:
            self.write_json_to_file(saliency_dict, self.base_dir, 'saliency_scores.json', prefix)
            self.write_json_to_file(epoch_metric_matrix, self.base_dir, 'epoch_metric_matrix.json', prefix)
            self.write_metrics_to_csv(self.base_dir, 'per_point_metrics.csv', prefix)

if __name__=='__main__':
    script = GradientScript()
    script.run()