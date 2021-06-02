from typing import Optional, Dict, Any, List
import math
import numpy as np
import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, Average

@Model.register('gradient_search_model')
class GradientSearchModel(Model):
    def __init__(
        self,
        cuda_device: int,
        vocab: Vocabulary,
        task_model: Model,
        full_predicted_label: int,
        full_predicted_prob: float,
        initial_logits: torch.tensor,
        sparsity: float = 0.05,
        sparsity_weight: float = 1,
        objective: str = 'suff',
        top_k_selection: str = 'exact_k',
        task_model_dropout: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):
        super(GradientSearchModel, self).__init__(vocab, regularizer)
        self.cuda_device = cuda_device

        self.vocab = vocab
        self.num_choices = self.vocab.get_vocab_size('labels')
        self.task_model = task_model
        assert initial_logits.dtype == torch.float32
        self.logits = torch.nn.Parameter(initial_logits, requires_grad=True)

        self.full_predicted_label = torch.tensor(full_predicted_label, device=self.cuda_device).unsqueeze(dim=0)
        self.full_predicted_prob = torch.tensor(full_predicted_prob, device=self.cuda_device)
        self.doc_length = initial_logits.size(0)

        self.sparsity = sparsity
        self.sparsity_weight = sparsity_weight
        self.sparsity_list = [0.05, 0.1, 0.2, 0.5]

        assert objective in ['comp', 'suff', 'both']
        self.objective = objective
        assert top_k_selection in ['exact_k', 'up_to_k']
        self.top_k_selection = top_k_selection
        self.task_model_dropout = task_model_dropout

        self.avg_task_loss = Average()
        self.avg_sparsity_loss = Average()
        self.avg_expected_sparsity = Average()
        self.avg_true_sparsity = Average()
        self.epoch_metric_dict = {}

        self.epoch = None

        initializer(self)

    def forward(self, **kwargs):
        if self.training:
            return self._forward(**kwargs)
        else:
            return self._validate(**kwargs)

    def _forward(self, document, always_keep_mask, label=None, metadata=None) -> Dict[str, Any]:
        if self.task_model_dropout:
            self.task_model.train()
        else:
            self.task_model.eval()

        num_samples = always_keep_mask.size(0)

        suff_mask = always_keep_mask.clone().float()
        comp_mask = always_keep_mask.clone().float()
        avg_true_sparsity = 0
        for i in range(num_samples):
            doc_mask = self._gumbel_softmax(self.logits)[..., 1]
            avg_true_sparsity += torch.sum(doc_mask).item()
            suff_mask[i, ..., 1:(self.doc_length+1)] = doc_mask
            comp_mask[i, ..., 1:(self.doc_length+1)] = (1 - doc_mask)

        batch_predicted_label = torch.cat([self.full_predicted_label for i in range(num_samples)], dim=0)

        task_loss = 0
        if self.objective in ['suff', 'both']:
            document['task']['mask'] = suff_mask
            suff_output_dict = self.task_model._forward(
                document, always_keep_mask, batch_predicted_label, metadata)
            task_loss += suff_output_dict['loss']
        if self.objective in ['comp', 'both']:
            document['task']['mask'] = comp_mask
            comp_output_dict = self.task_model._forward(
                document, always_keep_mask, batch_predicted_label, metadata)
            task_loss -= comp_output_dict['loss']

        # Sparsity loss
        mask_probs = torch.softmax(self.logits, dim=-1)[..., 1]
        expected_sparsity = torch.sum(mask_probs, dim=-1)
        target_sparsity = self.doc_length * self.sparsity
        sparsity_loss = torch.mean(torch.square(expected_sparsity - target_sparsity))
        avg_true_sparsity /= num_samples

        alpha = self.sparsity_weight / (1 + self.sparsity_weight)
        loss = (1 - alpha) * task_loss + alpha * sparsity_loss

        # Metrics
        self.avg_task_loss(task_loss.item())
        self.avg_sparsity_loss(sparsity_loss.item())
        self.avg_expected_sparsity(torch.mean(expected_sparsity).item())
        self.avg_true_sparsity(avg_true_sparsity)

        output_dict = {}
        output_dict['loss'] = loss

        return output_dict

    def _validate(self, 
                  document,
                  always_keep_mask,
                  label,
                  metadata=None) -> Dict[str, Any]:
        self.task_model.eval()
        with torch.no_grad():
            original_mask = document['task']['mask']
            positive_probs = torch.softmax(self.logits, dim=-1)[..., 1]

            # task loss across sparsity
            # task_loss = 0
            task_metric = 0
            for sparsity in self.sparsity_list:
                k = math.ceil(sparsity * self.doc_length)
                if self.top_k_selection == 'up_to_k':
                    num_probable_tokens = torch.sum(positive_probs > 0.5).item()
                    k = min(k, num_probable_tokens)

                # Validate task loss
                _, indices = torch.topk(positive_probs, k)
                if self.objective in ['suff', 'both']:
                    suff_mask = always_keep_mask.detach().clone().float()
                    suff_mask[..., (indices+1)] = 1
                    document['task']['mask'] = suff_mask
                    predicted_output_dict = self.task_model._forward(
                        document, always_keep_mask, self.full_predicted_label, metadata)
                    # task_loss += predicted_output_dict['loss']
                    task_metric += self.full_predicted_prob - predicted_output_dict['label_prob']
                if self.objective in ['comp', 'both']:
                    comp_mask = original_mask.detach().clone().float()
                    comp_mask[..., (indices+1)] = 0
                    document['task']['mask'] = comp_mask
                    predicted_output_dict = self.task_model._forward(
                        document, always_keep_mask, self.full_predicted_label, metadata)
                    # task_loss -= predicted_output_dict['loss']
                    task_metric -= self.full_predicted_prob - predicted_output_dict['label_prob']

        output_dict = {}
        output_dict['saliency'] = self.logits.detach().unsqueeze(dim=0)
        # output_dict['loss'] = task_loss / len(self.sparsity_list)
        output_dict['loss'] = task_metric / len(self.sparsity_list)

        if self.epoch is not None:
            self.epoch_metric_dict[self.epoch] = output_dict['loss']

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        class_metrics = {}

        # Training stats
        if self.training:
            class_metrics.update({'task_loss': self.avg_task_loss.get_metric(reset)})
            class_metrics.update({'sp_loss': self.avg_sparsity_loss.get_metric(reset)})
            class_metrics.update({'exp_sp': self.avg_expected_sparsity.get_metric(reset)})
            class_metrics.update({'true_sp': self.avg_true_sparsity.get_metric(reset)})

        return class_metrics

    def _gumbel_softmax(self, logits, temperature=1, eps=1e-20):
        '''logits: [batch_size, seq_length, 2]'''
        uniform = torch.rand(logits.size(), device=self.cuda_device)
        gumbel = -torch.autograd.Variable(torch.log(-torch.log(uniform + eps) + eps)).to(device=self.cuda_device)
        y = logits + gumbel
        y = torch.softmax(y / temperature, dim=-1)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y

        return y_hard

    def get_epoch_metric_dict(self):
        return self.epoch_metric_dict