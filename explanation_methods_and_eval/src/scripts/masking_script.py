import torch
import argparse
from overrides import overrides
import os
import time
import math
import sys
import shutil
import numpy as np

from allennlp.data import PyTorchDataLoader
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.trainer import GradientDescentTrainer

from src.scripts.base_script import BaseScript
from src.dataset_readers.masking_reader import MaskingReader
from src.models.masking_model import MaskingModel

class MaskingScript(BaseScript):
    @overrides
    def init_parser_args(self, parser):
        super().init_parser_args(parser)
        parser.add_argument('--objective', type=str)
        parser.add_argument('--top_k_selection', type=str, choices=['exact_k', 'up_to_k'])
        parser.add_argument('--sparsity', type=float)
        parser.add_argument('--sparsity_weight', type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--num_epochs', type=int)
        parser.add_argument('--patience', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--masking_model_name', type=str, default='bert-base-uncased')
        parser.add_argument('--masking_model_exp_name', type=str, default='default', help='Used to load non-default masking model')
        parser.add_argument('--use_gumbel', action='store_true')
        parser.add_argument('--n_choose_k', action='store_true')
        parser.add_argument('--eval_only', action='store_true')

    @overrides
    def init_base_dir(self, args):
        if not args.debug:
            self.base_dir = f'outputs/masking_model/{args.dataset}'
        else:
            self.base_dir = 'outputs/masking_model/debug'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def init_masking_reader(self, args):
        token_indexers = {}
        token_indexers['task'] = PretrainedTransformerIndexer(model_name=args.task_model_name)
        token_indexers['mask'] = PretrainedTransformerMismatchedIndexer(model_name=args.masking_model_name, max_length=512)
        tokenizer = PretrainedTransformerTokenizer(model_name=args.task_model_name)
        reader = MaskingReader(
            dataset=args.dataset, 
            token_indexers=token_indexers, 
            tokenizer=tokenizer, 
            num_datapoints=args.num_datapoints
        )

        return reader

    def init_masking_model(self, args, task_model, vocab):
        token_embedders = {
            'mask': PretrainedTransformerMismatchedEmbedder(model_name=args.masking_model_name, max_length=512)
        }
        text_field_embedder = BasicTextFieldEmbedder(token_embedders=token_embedders)
        masking_model = MaskingModel(
            cuda_device=args.cuda_device,
            vocab=vocab,
            task_model=task_model,
            text_field_embedder=text_field_embedder,
            objective=args.objective,
            top_k_selection=args.top_k_selection,
            sparsity_weight=args.sparsity_weight,
            sparsity=args.sparsity,
            sparsity_list=self.sparsity_list,
            use_gumbel=args.use_gumbel,
            n_choose_k=args.n_choose_k
        )
        masking_model.to(args.cuda_device)

        return masking_model

    def load_masking_model(self, args, masking_model):
        data_folder = args.dataset
        if args.dataset in ['cose_short']:
            data_folder = 'cose'

        weights_path = f'outputs/masking_model/{data_folder}/{args.masking_model_exp_name}/best.th'
        masking_model.load_state_dict(torch.load(weights_path))

        return masking_model

    def predict(self, args, masking_model, data_loader):
        saliency_dict = {}
        with torch.no_grad():
            masking_model.eval()
            iterator = iter(data_loader)
            generator_tqdm = Tqdm.tqdm(iterator)

            for batch in generator_tqdm:
                batch = nn_util.move_to_device(batch, args.cuda_device)
                instance_ids = [metadata['instance_id'] for metadata in batch['metadata']]
                output_dict = masking_model._validate(**batch)
                
                for i, instance_id in enumerate(instance_ids):
                    saliency_dict[instance_id] = output_dict['saliency'][i].tolist()

        return saliency_dict

    def train(self, args, masking_model, train_data_loader, validation_data_loader):
        parameters = [
            [n, p]
            for n, p in masking_model.named_parameters() if p.requires_grad
        ]
        parameter_groups = [(['task_model'], {'requires_grad': False})]
        optimizer = AdamOptimizer(parameters, parameter_groups, lr=args.lr)

        serialization_dir = os.path.join(self.base_dir, args.exp_name)
        if os.path.exists(serialization_dir):
            shutil.rmtree(serialization_dir)
        os.makedirs(serialization_dir, exist_ok=True)
        checkpointer = Checkpointer(serialization_dir=serialization_dir, num_serialized_models_to_keep=1)

        trainer = GradientDescentTrainer(
            model=masking_model, 
            data_loader=train_data_loader, 
            checkpointer=checkpointer,
            validation_data_loader=validation_data_loader, 
            validation_metric='-loss',
            num_epochs=args.num_epochs,
            patience=args.patience,
            optimizer=optimizer, 
            cuda_device=args.cuda_device,
            grad_norm=10.0
        )
        trainer.train()
        masking_model.load_state_dict(torch.load(os.path.join(serialization_dir, 'best.th')))

        return masking_model

    @overrides
    def run(self):
        args = self.initialize(enable_warnings=False)

        task_reader = self.init_task_reader(args)
        task_data, task_vocab = self.read_data(args, task_reader)
        task_model = self.init_task_model(args, task_vocab)
        task_model = self.load_task_model(args, task_model)

        masking_reader = self.init_masking_reader(args)
        train_data, train_vocab = self.read_data(args, masking_reader, 'train')
        val_data, val_vocab = self.read_data(args, masking_reader, 'val')
        train_data_loader = PyTorchDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        validation_data_loader = PyTorchDataLoader(val_data, batch_size=args.batch_size, shuffle=False)

        masking_model = self.init_masking_model(args, task_model, train_vocab)        

        if not args.eval_only:
            masking_model = self.train(args, masking_model, train_data_loader, validation_data_loader)
        else:
            masking_model = self.load_masking_model(args, masking_model)

        predict_data, predict_vocab = self.read_data(args, masking_reader)
        predict_data_loader = PyTorchDataLoader(predict_data, batch_size=args.batch_size, shuffle=False)
        # import pdb; pdb.set_trace()
        saliency_dict = self.predict(args, masking_model, predict_data_loader)

        with torch.no_grad():
            task_model.eval()
            task_data_loader = PyTorchDataLoader(task_data, batch_size=1, shuffle=False)
            iterator = iter(task_data_loader)
            generator_tqdm = Tqdm.tqdm(iterator)

            start = time.time()
            for step, batch in enumerate(generator_tqdm):
                batch = nn_util.move_to_device(batch, args.cuda_device)
                instance_id = batch['metadata'][0]['instance_id']
                saliency = torch.tensor(saliency_dict[instance_id], device=args.cuda_device)

                self.update_saliency_metrics(args, task_model, batch, saliency, print_examples = (step==0))
                result_dict, result_str = self.format_metrics()
                generator_tqdm.set_description(result_str, refresh=False)

            end = time.time()
            result_dict['time'] = end - start
            result_dict['args'] = str(sys.argv)
            print('-----Final Result-----')
            print('results: ', result_dict)
            print('-----------')

            output_dir = os.path.join(self.base_dir, args.datasplit)
            prefix = args.exp_name
            self.write_json_to_file(result_dict, self.base_dir, 'final_result.json', prefix)
            if not args.eval_only:
                self.write_json_to_file(saliency_dict, self.base_dir, 'saliency_scores.json', prefix)
            self.write_metrics_to_csv(self.base_dir, 'per_point_metrics.csv', prefix)

if __name__=='__main__':
    script = MaskingScript()
    script.run()