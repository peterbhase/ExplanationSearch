from typing import Optional, Dict, Any
import math
import torch
import torch.nn.functional as F

from allennlp.data import Batch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.attention import Attention
from allennlp.training.metrics import CategoricalAccuracy

@Model.register('task_model')
class TaskModel(Model):
    def __init__(
        self,
        cuda_device: int,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        is_multiple_choice: bool = False
    ):
        super(TaskModel, self).__init__(vocab, regularizer)
        self.cuda_device = cuda_device

        self.vocab = vocab
        self.num_labels = self.vocab.get_vocab_size('labels')
        # assert self.num_labels > 1

        self.text_field_embedder = text_field_embedder
        self.transformer_model = self.text_field_embedder._token_embedders['task'].transformer_model
        self.embedder_hidden_size = self.transformer_model.config.hidden_size

        self.is_multiple_choice = is_multiple_choice

        if self.num_labels > 1 and not self.is_multiple_choice:
            classification_output_dim = self.num_labels
        elif self.num_labels > 1 and self.is_multiple_choice:
            classification_output_dim = 1
        elif self.num_labels <= 1:
            classification_output_dim = 2 # assume binary task
        self.classification_layer = torch.nn.Linear(self.embedder_hidden_size, classification_output_dim)

        self.accuracy = CategoricalAccuracy()

        initializer(self)

    def forward(self, **kwargs):
        return self._forward(**kwargs)

    def _forward(self, document, always_keep_mask=None, label=None, metadata=None, one_hot=False, ratio=1, baseline='zero',
                    one_hot_input=None, inputs_embeds=None) -> Dict[str, Any]:
        '''
        only one of document, one_hot_input, and inputs_embeds should be not None
        - one_hot_input can actually be continuous in id space. for use with integrated gradients
        '''
        if self.is_multiple_choice:
            self._unfold_document(document)  # Reshape input document

        if one_hot_input is None and inputs_embeds is None:
            assert document is not None
            if one_hot:
                token_ids_one_hot, pooled_embedding = self._one_hot_get_embedding(document, metadata, ratio, baseline)
            else:
                _, pooled_embedding = self.text_field_embedder(document)

        if one_hot_input is not None:
            batch_size = one_hot_input.size(0)
            embedding_layer = self.transformer_model.get_input_embeddings()
            input_embeddings = torch.cat([torch.arange(self.transformer_model.config.vocab_size).unsqueeze(dim=0) for _ in range(batch_size)], dim=0).to(self.cuda_device)
            input_embeddings = embedding_layer(input_embeddings) # batch_size, vocab_size, hidden_dim
            inputs_embeds = torch.bmm(one_hot_input, input_embeddings) # batch_size, seq_len, hidden_dim

        if inputs_embeds is not None:
            transformer_output = self.transformer_model.forward(
                attention_mask=document['task']['mask'],
                token_type_ids=document['task']['type_ids'],
                inputs_embeds=inputs_embeds)
            pooled_embedding = transformer_output[1]

        logits = self.classification_layer(pooled_embedding)
        if self.is_multiple_choice:
            logits = logits.view(-1, self.num_labels)
        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}
        output_dict['logits'] = logits
        output_dict['probs'] = probs
        output_dict['predicted_prob'] = probs.max(-1)[0]
        output_dict['predicted_label'] = probs.argmax(-1)
        if one_hot:
            output_dict['token_ids_one_hot'] = token_ids_one_hot

        if label is not None:
            loss = F.cross_entropy(logits, label, reduction='none')
            loss_mean = loss.mean()
            output_dict['loss'] = loss_mean
            output_dict['batch_loss'] = loss
            output_dict['label'] = label
            output_dict['label_prob'] = probs[torch.arange(probs.size(0)), label]
            self.accuracy(output_dict['logits'], output_dict['label'])

        if self.is_multiple_choice:
            self._fold_document(document)  # Reshape back

        return output_dict

    def _one_hot_get_ids(self, document, metadata=None, ratio=1, baseline='mask'):
        batch_size = document['task']['token_ids'].size(0)
        vocab_size = self.transformer_model.config.vocab_size

        token_ids_one_hot = torch.nn.functional.one_hot(
            document['task']['token_ids'], num_classes=vocab_size).float() # batch_size, seq_len, vocab_size

        if ratio != 1:
            assert metadata is not None
            assert len(metadata) == 1, 'Do not support batch yet.'
            assert baseline in ['zero', 'mask']
            assert ratio >= 0 and ratio <= 1
            doc_length = len(metadata[0]['doc_tokens'])
            
            if baseline == 'zero':
                baseline_one_hot = token_ids_one_hot.detach().clone()
                baseline_one_hot[..., 1:(doc_length+1), :] = 0
            elif baseline == 'mask':
                masked_token_ids = document['task']['token_ids'].detach().clone()
                masked_token_ids[..., 1:(doc_length+1)] = metadata[0]['mask_token_id']
                baseline_one_hot = torch.nn.functional.one_hot(masked_token_ids, num_classes=vocab_size).float()
            
            token_ids_one_hot = ratio * token_ids_one_hot + (1 - ratio) * baseline_one_hot

        token_ids_one_hot.requires_grad_()
        return token_ids_one_hot

    def _one_hot_get_embedding(self, document, metadata=None, ratio=1, baseline='zero', return_inputs_embeds=False):
        batch_size = document['task']['token_ids'].size(0)
        vocab_size = self.transformer_model.config.vocab_size
        token_ids_one_hot = self._one_hot_get_ids(document, metadata=metadata, ratio=ratio, baseline=baseline)

        embedding_layer = self.transformer_model.get_input_embeddings()
        input_embeddings = torch.cat([torch.arange(vocab_size).unsqueeze(dim=0) for _ in range(batch_size)], dim=0).to(self.cuda_device)
        input_embeddings = embedding_layer(input_embeddings) # batch_size, vocab_size, hidden_dim
        input_embeddings = torch.bmm(token_ids_one_hot, input_embeddings) # batch_size, seq_len, hidden_dim

        if return_inputs_embeds:
            return token_ids_one_hot, input_embeddings

        else:
            transformer_output = self.transformer_model.forward(
                attention_mask=document['task']['mask'],
                token_type_ids=document['task']['type_ids'],
                inputs_embeds=input_embeddings)
            pooled_embedding = transformer_output[1]
            return token_ids_one_hot, pooled_embedding

    def _unfold_document(self, document):
        for k, v in document['task'].items():
            if len(v.size()) == 3:
                document['task'][k] = v.view(-1, v.size(dim=-1))
            elif len(v.size()) == 4:
                document['task'][k] = v.view(-1, v.size(dim=2), 2)

    def _fold_document(self, document):
        for k, v in document['task'].items():
            if len(v.size()) == 2:
                document['task'][k] = v.view(-1, self.num_labels, v.size(dim=-1))
            if len(v.size()) == 3:
                document['task'][k] = v.view(-1, self.num_labels, v.size(dim=1), 2)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        class_metrics = {}
        class_metrics['accuracy'] = self.accuracy.get_metric(reset)
        
        return class_metrics