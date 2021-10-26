from typing import Dict, List
import argparse
import math
import statistics
import torch
import os
import json
import copy
import warnings
import logging
import sys
import numpy as np
import pandas as pd
import time
import operator as op
from functools import reduce

from allennlp.data import Vocabulary
from allennlp.data import PyTorchDataLoader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.common.checks import check_for_gpu
from allennlp.training.metrics import Average, CategoricalAccuracy
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

from src.dataset_readers.task_reader import TaskReader
from src.models.task_model import TaskModel

class FidelityMetrics:
    '''Data structure to hold the average accuracy and suff/comp for all datapoints.'''
    def __init__(self):
        self.acc = CategoricalAccuracy()
        self.metrics = []
        self.pred_probs = []
        self.woes = []
        self.woe_diffs = []

    def stdev(self, statistic='metrics'):
        quantity = getattr(self, statistic)
        if len(quantity) < 2:
            return 0
        else:
            return statistics.stdev(quantity) / math.sqrt(len(quantity))

    def average(self):
        return statistics.mean(self.metrics)

    def accuracy(self):
        return self.acc.get_metric()

    def get_weight_of_evidence(self):
        return np.mean(self.woe_diffs)

    def to_dict(self):
        return {
            'avg': self.average(), 
            'stdev': self.stdev(statistic='metrics'),
            'acc': self.accuracy(),
            'woe': self.get_weight_of_evidence(),
            'woe_stdev' : self.stdev(statistic='woe_diffs')
        }

    def is_empty(self):
        return len(self.metrics) == 0

class BestMetric:
    '''Data structure to hold the best suff/comp found during search.'''
    def __init__(self, objective: str):
        assert objective in ['min', 'max']
        self.objective = objective
        self.best_value = float('inf') if self.objective == 'min' else float('-inf')
        self.best_logits = None
        self.best_label_prob = None
        self.best_doc_mask = None
        self.all_metrics = []
        self.all_metrics_woe = []

    def np_woe(self, p):
        return np.log(p / (1-p))

    def from_output_dict(self, output_dict, index, mask_list=None, save_metric=False, true_batch_size=None):
        current_value = output_dict['batch_loss'][index]
        if (self.objective == 'min' and current_value < self.best_value) or (self.objective == 'max' and current_value > self.best_value):
            self.best_value = output_dict['batch_loss'][index]
            self.best_logits = output_dict['logits'][index, ...].unsqueeze(dim=0)
            self.best_label_prob = output_dict['label_prob'][index]
            if mask_list:
                self.best_doc_mask = mask_list[index]
        if save_metric:
            # these get turned into metric and metric_woe in update_search_metrics
            if true_batch_size:
                output_dict['label_prob'] = output_dict['label_prob'][:true_batch_size]
            self.all_metrics.extend(output_dict['label_prob'].tolist())
            self.all_metrics_woe.extend([self.np_woe(prob) for prob in output_dict['label_prob'].tolist()])
            self.all_metrics_best_so_far = []
            self.all_metrics_best_so_far_woe = []


class BaseScript:
    def init_parser_args(self, parser):
        parser.add_argument('--cuda_device', type=int)
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--datasplit', type=str, choices=['train', 'val', 'test'])
        parser.add_argument('--num_datapoints', type=int)
        parser.add_argument('--seed', type=int, default=200)
        parser.add_argument('--exp_name', type=str)
        parser.add_argument('--task_model_name', type=str, default='bert-base-uncased')
        parser.add_argument('--task_model_exp_name', type=str, default='default', help='Used to load non-default task model')
        parser.add_argument('--debug', action='store_true')

    def check_args(self, args):
        pass

    def init_torch(self, args):
        check_for_gpu(args.cuda_device)
        torch.cuda.set_device(args.cuda_device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    def init_base_dir(self, args):
        raise NotImplementedError

    def weight_of_evidence(self, p):
        return torch.log(p / (1-p))

    def np_woe(self, p):
        return np.log(p / (1-p))

    def initialize(self, enable_logging=False, enable_warnings=True):
        if enable_logging:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        if not enable_warnings:
            warnings.filterwarnings('ignore', category=FutureWarning)

        parser = argparse.ArgumentParser()
        self.init_parser_args(parser)
        args = parser.parse_args()
        self.check_args(args)
        self.init_torch(args)
        self.init_base_dir(args)

        self.sparsity_list = [0.05, 0.1, 0.2, 0.5]
        self.metrics_dict = {
            'suff': {sparsity: FidelityMetrics() for sparsity in ([0] + self.sparsity_list)},
            'comp': {sparsity: FidelityMetrics() for sparsity in ([0] + self.sparsity_list)}
        }

        return args

    def init_task_reader(self, args):
        tokenizer = PretrainedTransformerTokenizer(model_name=args.task_model_name)
        token_indexers = {'task': PretrainedTransformerIndexer(model_name=args.task_model_name)}
        reader = TaskReader(
            dataset=args.dataset, 
            token_indexers=token_indexers, tokenizer=tokenizer, 
            num_datapoints=args.num_datapoints
        )

        return reader

    def read_data(self, args, reader, datasplit=None):
        '''If datasplit is provided, use datasplit. Otherwise, default to args.datasplit.'''
        datasplit = datasplit if datasplit else args.datasplit
        data_path = os.path.join('data', args.dataset, f'{datasplit}.jsonl')
        data = reader.read(data_path)
        vocab = Vocabulary.from_instances(data)
        data.index_with(vocab)

        return data, vocab

    def init_task_model(self, args, vocab):
        token_embedders = {'task': PretrainedTransformerEmbedder(model_name=args.task_model_name)}
        text_field_embedder = BasicTextFieldEmbedder(token_embedders=token_embedders)
        is_multiple_choice = (args.dataset in ['cose_short', 'cose'])
        task_model = TaskModel(
            cuda_device=args.cuda_device,
            vocab=vocab, 
            text_field_embedder=text_field_embedder, 
            is_multiple_choice=is_multiple_choice
        )
        task_model.to(args.cuda_device)

        return task_model

    def load_task_model(self, args, task_model, use_last=False):
        if args.dataset in ['cose_short']:
            data_folder = 'cose'
        else:
            data_folder = args.dataset
        name = 'best' if not use_last else 'model_state_epoch_4'
        weights_path = f'outputs/task_model/{data_folder}/{args.task_model_exp_name}/{name}.th'
        task_model.load_state_dict(torch.load(weights_path))

        return task_model

    def update_saliency_metrics(self, args, task_model, batch, saliency: torch.Tensor, print_examples:bool = False):
        '''Calculate suff / comp for saliency-based methods: gradient search, lime, masking model'''
        document = batch['document']
        metadata = batch['metadata'][0]
        always_keep_mask = batch['always_keep_mask']
        original_mask = document['task']['mask']
        label = batch['label']

        full_output_dict = task_model._forward(document=document, label=label)
        full_predicted_label = full_output_dict['predicted_label']

        assert len(saliency.size()) == 2 and saliency.size(1) == 2, 'saliency should be tensor of size (doc_length, 2)'
        assert saliency.size(0) == len(metadata['doc_tokens'])
        doc_length = saliency.size(0)
        positive_probs = torch.softmax(saliency, dim=-1)[..., 1]

        if print_examples:
            print("Example data point: ")
            tokens = batch['metadata'][0]['doc_tokens']
            print_seq = " ".join([str(token) for token in tokens])
            print(f"Sequence: {print_seq}")
            for i in range(len(tokens)):
                print(f" {tokens[i]} : {positive_probs[i].item():.2f}")

        for sparsity in ([0] + self.sparsity_list):
            k = math.ceil(doc_length * sparsity)
            if args.top_k_selection == 'up_to_k':
                num_positive_scores = torch.sum(positive_probs > .5).item()
                k = min(k, num_positive_scores)

            suff_mask = always_keep_mask.detach().clone().float()
            comp_mask = original_mask.detach().clone().float()
            if sparsity:
                _, indices = torch.topk(positive_probs, k, dim=-1)
                suff_mask[..., indices+1] = 1
                comp_mask[..., indices+1] = 0

            with torch.no_grad():
                document['task']['mask'] = suff_mask
                suff_output_dict = task_model._forward(document=document, label=full_predicted_label)
                self.metrics_dict['suff'][sparsity].acc(suff_output_dict['logits'], label)
                self.metrics_dict['suff'][sparsity].metrics.append(
                    (full_output_dict['predicted_prob'] - suff_output_dict['label_prob']).item()
                )
                self.metrics_dict['suff'][sparsity].woe_diffs.append(
                    (self.weight_of_evidence(full_output_dict['predicted_prob']) - self.weight_of_evidence(suff_output_dict['label_prob'])).item()
                )
                self.metrics_dict['suff'][sparsity].pred_probs.append(suff_output_dict['label_prob'].item())

                document['task']['mask'] = comp_mask
                comp_output_dict = task_model._forward(document=document,label=full_predicted_label)
                self.metrics_dict['comp'][sparsity].acc(comp_output_dict['logits'], label)
                self.metrics_dict['comp'][sparsity].metrics.append(
                    (full_output_dict['predicted_prob'] - comp_output_dict['label_prob']).item()
                )
                self.metrics_dict['comp'][sparsity].woe_diffs.append(
                    (self.weight_of_evidence(full_output_dict['predicted_prob']) - self.weight_of_evidence(comp_output_dict['label_prob'])).item()
                )
                self.metrics_dict['comp'][sparsity].pred_probs.append(comp_output_dict['label_prob'].item())

            if print_examples:
                if 'objective' not in args.__dict__.keys():
                    use_mask = suff_mask
                    # print(f"Suff mask ({sparsity}): {use_mask.tolist()}")
                else:
                    use_mask = suff_mask if args.objective == 'suff' else comp_mask
                    # print(f"{args.objective} mask ({sparsity}): {use_mask.tolist()}")
                print_seq = " ".join([str(token) if use_mask[0, i]==1 else '__' for i, token in enumerate(tokens)])
                print(f"Kept tokens: {print_seq}")

    def update_search_metrics(self, args, task_model, batch, sparsity, mask_list, initial_mask=None, objective='both'):
        '''Calculate suff / comp for search-based methods: exhaustive, ordered. (Hotflip is weird)'''
        full_output_dict = task_model._forward(**batch)
        if args.batch_size > len(mask_list):
            stack_batch_size = len(mask_list)
        else:
            stack_batch_size = args.batch_size
        batch_document, batch_predicted_label = \
            self.stack_input(batch['document'], full_output_dict['predicted_label'], stack_batch_size)
        doc_length = len(batch['metadata'][0]['doc_tokens'])

        min_suff = BestMetric(objective='min')
        max_comp = BestMetric(objective='max')
        if hasattr(args, 'save_all_metrics'):
            save_all_metrics = args.save_all_metrics
        else:
            save_all_metrics = False

        forward_times = 0

        # assert args.batch_size > 1
        if initial_mask is not None:
            batch_document['task']['mask'][0, ..., 1:(doc_length+1)] = initial_mask
            batch_document['task']['mask'][1, ..., 1:(doc_length+1)] = 1 - initial_mask
            output_dict = task_model._forward(
                batch_document, label=batch_predicted_label)
            min_suff.from_output_dict(output_dict, 0, mask_list=[initial_mask.tolist()])
            max_comp.from_output_dict(output_dict, 1, mask_list=[None, (1-initial_mask).tolist()])

        if len(mask_list) < args.batch_size:
            iter_idx = list(range(1))
        else:
            iter_idx = list(range(0, len(mask_list), args.batch_size))

        for i in iter_idx:
            true_batch_size = min(len(mask_list) - i, args.batch_size)

            # Sufficiency
            if objective in ['suff', 'both']:
                suff_doc_masks = []
                for j in range(true_batch_size):
                    suff_doc_mask_tensor = torch.tensor(mask_list[i+j])
                    batch_document["task"]["mask"][j, ..., 1:(doc_length+1)] = suff_doc_mask_tensor
                    suff_doc_masks.append(suff_doc_mask_tensor.tolist())
                suff_output_dict = task_model._forward(
                    document=batch_document, label=batch_predicted_label)
                batch_loss = suff_output_dict['batch_loss'][:true_batch_size]
                _, suff_index = batch_loss.min(dim=0)
                min_suff.from_output_dict(suff_output_dict, suff_index, mask_list=suff_doc_masks, 
                    save_metric=save_all_metrics, true_batch_size=true_batch_size)
                # print(suff_doc_mask_tensor)
                # print("per point metric: ", full_output_dict['predicted_prob'] - suff_output_dict['label_prob'])
                # print("best suff: ", full_output_dict['predicted_prob'] - min_suff.best_label_prob)

            if objective in ['comp', 'both']:
                # Comprehensiveness
                comp_doc_masks = []
                for j in range(true_batch_size):
                    if not args.eval_only:
                        comp_doc_mask_tensor = 1 - torch.tensor(mask_list[i+j])
                    else:
                        comp_doc_mask_tensor = torch.tensor(mask_list[i+j])
                    batch_document["task"]["mask"][j, ..., 1:(doc_length+1)] = comp_doc_mask_tensor
                    comp_doc_masks.append(comp_doc_mask_tensor.tolist())
                comp_output_dict = task_model._forward(
                    document=batch_document, label=batch_predicted_label)
                batch_loss = comp_output_dict['batch_loss'][:true_batch_size]
                _, comp_index = batch_loss.max(dim=0)
                max_comp.from_output_dict(comp_output_dict, comp_index, mask_list=comp_doc_masks, 
                    save_metric=save_all_metrics, true_batch_size=true_batch_size)                

        # update suff metrics
        if objective in ['suff', 'both']:
            self.metrics_dict['suff'][sparsity].acc(min_suff.best_logits, batch['label'])
            self.metrics_dict['suff'][sparsity].metrics.append(
                (full_output_dict['predicted_prob'] - min_suff.best_label_prob).item()
            )
            self.metrics_dict['suff'][sparsity].woe_diffs.append(
                (self.weight_of_evidence(full_output_dict['predicted_prob']) - self.weight_of_evidence(min_suff.best_label_prob)).item()
            )
            self.metrics_dict['suff'][sparsity].pred_probs.append(min_suff.best_label_prob.item())
            if save_all_metrics:
                best_so_far = 2e8                
                best_so_far_woe = 2e8
                for i in range(len(min_suff.all_metrics)):
                    metric = full_output_dict['predicted_prob'].item() - min_suff.all_metrics[i]
                    metric_woe = self.np_woe(full_output_dict['predicted_prob'].item()) - min_suff.all_metrics_woe[i]
                    min_suff.all_metrics[i] = metric
                    min_suff.all_metrics_woe[i] = metric_woe
                    if metric < best_so_far:
                        best_so_far=metric
                        best_so_far_woe=metric_woe
                    min_suff.all_metrics_best_so_far.append(best_so_far)
                    min_suff.all_metrics_best_so_far_woe.append(best_so_far_woe)

            # print("RS best: ")
            # print(min_suff.best_doc_mask)
            # print("suff: ", full_output_dict['predicted_prob'] - min_suff.best_label_prob)
            
        # update comp metrics
        if objective in ['comp', 'both']:
            self.metrics_dict['comp'][sparsity].acc(max_comp.best_logits, batch['label'])
            self.metrics_dict['comp'][sparsity].metrics.append(
                (full_output_dict['predicted_prob'] - max_comp.best_label_prob).item()
            )
            self.metrics_dict['comp'][sparsity].woe_diffs.append(
                (self.weight_of_evidence(full_output_dict['predicted_prob']) - self.weight_of_evidence(max_comp.best_label_prob)).item()
            )
            self.metrics_dict['comp'][sparsity].pred_probs.append(max_comp.best_label_prob.item())
            if save_all_metrics:
                best_so_far = -2e8                
                best_so_far_woe = -2e8
                for i in range(len(max_comp.all_metrics)):
                    metric = full_output_dict['predicted_prob'].item() - max_comp.all_metrics[i]
                    metric_woe = self.np_woe(full_output_dict['predicted_prob'].item()) - max_comp.all_metrics_woe[i]
                    max_comp.all_metrics[i] = metric
                    max_comp.all_metrics_woe[i] = metric_woe
                    if metric > best_so_far:
                        best_so_far=metric
                        best_so_far_woe=metric_woe
                    max_comp.all_metrics_best_so_far.append(best_so_far)
                    max_comp.all_metrics_best_so_far_woe.append(best_so_far_woe)
        # print(f"base forward time {(forward_times):.2f} seconds")
        return min_suff, max_comp

    def read_saliency(self, args, saliency_exp_name=None):
        if saliency_exp_name is None: saliency_exp_name = args.exp_name
        saliency_filepath = os.path.join(
            'outputs', args.initialization, args.dataset, args.datasplit, f'{saliency_exp_name}_saliency_scores.json')
        with open(saliency_filepath) as input_file:
            saliency_dict = json.load(input_file)
        return saliency_dict

    def write_json_to_file(self, object, output_dir, filename, prefix=''):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if '.json' not in filename:
            filename += '.json'
        if prefix and prefix[-1] != '_':
            prefix += '_'
        filename = prefix + filename
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as output_file:
            json.dump(object, output_file)

    def write_trajectories_to_csv(self, args, trajectories, output_dir, filename, prefix=''):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if '.csv' not in filename:
            filename += '.csv'
        if prefix and prefix[-1] != '_':
            prefix += '_'
        filename = prefix + filename
        filepath = os.path.join(output_dir, filename)
        # cols = {'id' : [], 'sparsity' : [], 'step' : [], 'suff' : [], 'comp' : [], 'suff_woe' : [], 'comp_woe' : []}
        # data = pd.DataFrame(cols)
        # unpack trajectories
        all_suff_metrics_dict = trajectories['suff']
        all_suff_woe_metrics_dict = trajectories['suff_woe']
        all_comp_metrics_dict = trajectories['comp']
        all_comp_woe_metrics_dict = trajectories['comp_woe']
        num_points = args.num_datapoints
        data_dict = {}
        row_counter = 0
        for i in range(num_points):
            for sparsity in self.sparsity_list:
                num_searched = len(all_suff_metrics_dict[sparsity][i]) # not always equal to args.num_to_search, e.g. with SA restarts that divide the total budget
                for step in range(num_searched):
                    data_point = {
                        'id' : i,
                        'sparsity' : sparsity,
                        'step' : step,
                        'suff' : all_suff_metrics_dict[sparsity][i][step],
                        'comp' : all_comp_metrics_dict[sparsity][i][step],
                        'suff_woe' : all_suff_woe_metrics_dict[sparsity][i][step],
                        'comp_woe' : all_comp_woe_metrics_dict[sparsity][i][step],
                    }
                    data_dict[row_counter] = data_point
                    row_counter += 1
        data = pd.DataFrame.from_dict(data_dict, "index")
        data.to_csv(filepath, index=False)

    def write_metrics_to_csv(self, output_dir, filename, prefix=''):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if '.csv' not in filename:
            filename += '.csv'
        if prefix and prefix[-1] != '_':
            prefix += '_'
        filename = prefix + filename
        filepath = os.path.join(output_dir, filename)
        cols = {'id' : [], 'suff' : [], 'comp' : [], 'suff_woe' : [], 'comp_woe' : []}
        data = pd.DataFrame(cols)
        for metric in ['suff', 'comp']:
            raw_scores_list = []
            woes_list = []
            pred_probs_list = []
            for sparsity in self.sparsity_list:
                metrics_dict = self.metrics_dict[metric][sparsity]
                raw_scores = metrics_dict.metrics
                woes = metrics_dict.woe_diffs
                pred_probs = metrics_dict.pred_probs
                raw_scores_list.append(raw_scores)
                woes_list.append(woes)
                pred_probs_list.append(pred_probs)
            raw_scores = np.array(raw_scores_list)
            woes = np.array(woes_list)
            pred_probs = np.array(pred_probs_list)
            raw_scores = np.mean(raw_scores, axis=0)
            woes = np.mean(woes, axis=0)
            pred_probs = np.mean(pred_probs, axis=0)
            # update data
            idx = np.arange(len(woes))
            data['id'] = idx
            data[metric] = raw_scores
            data[f"{metric}_woe"] = woes
            data[f'{metric}_pred_prob'] = pred_probs
        data.to_csv(filepath, index=False)

    def write_fidelity_vector(self, output_dir, prefix=''):
        '''Write individual suff/comp values for each data point.'''
        for metric in ['suff', 'comp']:
            matrix = []
            for sparsity in self.sparsity_list:
                matrix.append(self.metrics_dict[metric][sparsity].metrics)
            matrix = np.array(matrix)
            avg_metric = np.mean(matrix, axis=0).tolist()
            self.write_json_to_file(avg_metric, output_dir, f'{metric}_vector.json', prefix)

    def format_metrics(self):
        result_dict = {}
        result_str_list = []
        for key in ['suff', 'comp']:
            result_dict[key] = {}

            # Results for each sparsity
            for sparsity, metrics in self.metrics_dict[key].items():
                if not metrics.is_empty():
                    result_dict[key][sparsity] = metrics.to_dict()

            # Average metrics for only sparsities in sparsity list
            if result_dict[key]:
                for stat in ['avg', 'stdev', 'acc', 'woe', 'woe_stdev']:
                    stat_list = [result_dict[key][sparsity][stat] for sparsity in self.sparsity_list]
                    result_dict[key][stat] = statistics.mean(stat_list) 

                result_str_list.append(
                    '{}: {:.3f} +- {:.3f}'.format(key, result_dict[key]['avg'], result_dict[key]['stdev'])
                )
                result_str_list.append(
                    '{}_woe: {:.3f} +- {:.3f}'.format(key, result_dict[key]['woe'], result_dict[key]['woe_stdev'])
                )
                result_str_list.append(
                    '{}_acc: {:.3f}'.format(key, result_dict[key]['acc'])
                )

        result_str = ', '.join(result_str_list)
        return result_dict, result_str

    def stack_input(self, document, label, batch_size):
        '''Vertically stack input batch_size number of times.'''
        batch_document = copy.deepcopy(document)
        for k, v in document['task'].items():
            batch_document['task'][k] = torch.cat([document['task'][k] for i in range(batch_size)], dim=0)
        batch_label = torch.cat([label for i in range(batch_size)], dim=0)
        return batch_document, batch_label

    def run(self):
        raise NotImplementedError

    def ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer // denom  # or / in Python 2