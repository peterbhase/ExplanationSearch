import json
import os
import torch
import shutil
import time
import sys
from overrides import overrides

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.training.util import evaluate
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.data import PyTorchDataLoader
from allennlp.models.archival import Archive

from src.models.task_model import TaskModel
from src.scripts.base_script import BaseScript


class TaskScript(BaseScript):
    '''Script to train the task model. See task_commands.txt for console commands.'''
    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--num_epochs', type=int)
        parser.add_argument('--patience', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--eval_only', action='store_true')
        parser.add_argument('--masking_augmentation', action='store_true')
        parser.add_argument('--PLS_masks', action='store_true')

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/task_model/{args.dataset}'
        else:
            self.base_dir = 'outputs/task_model/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def train(self, args, task_model, train_data_loader, validation_data_loader):
        parameters = [
            [n, p]
            for n, p in task_model.named_parameters() if p.requires_grad
        ]
        optimizer = AdamOptimizer(model_parameters=parameters, lr=args.lr)

        serialization_dir = os.path.join(self.base_dir, args.exp_name)
        if os.path.exists(serialization_dir):
            shutil.rmtree(serialization_dir)
        os.makedirs(serialization_dir, exist_ok=True)
        checkpointer = Checkpointer(serialization_dir=serialization_dir, num_serialized_models_to_keep=1)

        trainer = GradientDescentTrainer(
            model=task_model, 
            data_loader=train_data_loader, 
            checkpointer=checkpointer,
            validation_data_loader=validation_data_loader, 
            validation_metric='+accuracy',
            num_epochs=args.num_epochs,
            patience=args.patience,
            optimizer=optimizer, 
            cuda_device=args.cuda_device,
            grad_norm=100.,
            grad_clipping=None,
            masking_augmentation=args.masking_augmentation,
            PLS_masks=args.PLS_masks
        )
        trainer.train()
        task_model.load_state_dict(torch.load(os.path.join(serialization_dir, 'best.th')))

        return task_model

    @overrides
    def run(self):
        args = self.initialize(enable_logging=True)

        task_reader = self.init_task_reader(args)
        train_data, train_vocab = self.read_data(args, task_reader, datasplit='train')
        val_data, val_vocab = self.read_data(args, task_reader, datasplit='val')
        test_data, test_vocab = self.read_data(args, task_reader, datasplit='test')
        train_data_loader = PyTorchDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        validation_data_loader = PyTorchDataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_data_loader = PyTorchDataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        args.task_model_exp_name = args.exp_name
        task_model = self.init_task_model(args, train_vocab)

        start = time.time()
        if args.eval_only:
            task_model = self.load_task_model(args, task_model)
        else:
            task_model = self.train(args, task_model, train_data_loader, validation_data_loader)
        
        dev_metrics = evaluate(task_model, validation_data_loader, cuda_device=args.cuda_device)
        test_metrics = evaluate(task_model, test_data_loader, cuda_device=args.cuda_device)
        end = time.time()

        test_metrics['time'] = end - start
        test_metrics['args'] = str(sys.argv)

        print('-----Final Result-----')
        print(' Dev:')
        for k, v in dev_metrics.items():
            print(f"  {k} : {v}")
        print(" Test: ")
        for k, v in test_metrics.items():
            print(f"  {k} : {v}")
        print('----------')

        output_dir = os.path.join(self.base_dir, args.exp_name)
        self.write_json_to_file(dev_metrics, output_dir, 'dev_metrics.json')
        self.write_json_to_file(test_metrics, output_dir, 'test_metrics.json')

if __name__ == '__main__':
    script = TaskScript()
    script.run()
