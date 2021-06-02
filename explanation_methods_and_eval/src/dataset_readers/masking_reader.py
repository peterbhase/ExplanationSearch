from typing import List, Union, Dict
from overrides import overrides
import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.fields import LabelField, MetadataField, TextField, SequenceLabelField, ListField, ArrayField
from allennlp.data.instance import Instance

from src.dataset_readers.base_reader import BaseReader

@DatasetReader.register('masking_reader')
class MaskingReader(BaseReader):
    def __init__(
        self,
        dataset: str,
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Tokenizer,
        num_datapoints: int = None,
        lazy: bool = False
    ) -> None:
        super().__init__(dataset, token_indexers, tokenizer, num_datapoints, lazy)
        self.task_indexer = {"task": token_indexers["task"]}
        self.mask_indexer = {"mask": token_indexers["mask"]}

    def _task_tokens(self, doc_tokens, query_tokens, cls_token, sep_token):
        return [cls_token] + doc_tokens + [sep_token] + query_tokens + [sep_token]

    def _mask_tokens(self, doc_tokens, cls_token, sep_token):
        return [cls_token] + doc_tokens + [sep_token]

    def _always_keep_mask(self, doc_tokens, query_tokens):
        return [1] + [0] * len(doc_tokens) + [1] * (len(query_tokens) + 2)

    def _metadata(self, instance_id, doc_tokens, query_tokens):
        metadata = {
            "instance_id": instance_id,
            "doc_tokens": doc_tokens,
            "query_tokens": query_tokens,
        }
        return metadata

    @overrides
    def text_to_instance(
        self,
        instance_id: str,
        document: Union[str, List[str]],
        query: str,
        label: str = None,
    ) -> Instance:
        if isinstance(document, str):
            doc_tokens = self.tokenizer.tokenize(document)
        else:
            doc_tokens, _ = self.tokenizer.intra_word_tokenize(document)
        cls_token = doc_tokens[0]
        sep_token = doc_tokens[-1]
        doc_tokens = doc_tokens[1:-1]

        # Multiple choice case
        if self.is_multiple_choice:
            answer_choices = query.split("[sep]")
            query_tokens_list = [self.tokenizer.tokenize(choice)[1:-1] for choice in answer_choices]

            task_tokens_list = [
                self._task_tokens(doc_tokens, query_tokens, cls_token, sep_token)
                    for query_tokens in query_tokens_list
            ]
            mask_tokens = self._mask_tokens(doc_tokens, cls_token, sep_token)
            always_keep_mask_list = [
                self._always_keep_mask(doc_tokens, query_tokens)
                    for query_tokens in query_tokens_list
            ]

            fields = {}
            fields["task_document"] = ListField(
                [TextField(task_tokens, self.task_indexer) for task_tokens in task_tokens_list]
            )
            fields["mask_document"] = TextField(mask_tokens, self.mask_indexer)
            fields["always_keep_mask"] = ListField(
                [ArrayField(np.array(always_keep_mask)) for always_keep_mask in always_keep_mask_list]
            )
            fields["metadata"] = MetadataField(self._metadata(instance_id, doc_tokens, query_tokens_list))
            if label is not None:
                fields["label"] = LabelField(label, label_namespace="labels")

            return Instance(fields)

        # Binary classification case
        else:
            query_tokens = self.tokenizer.tokenize(query)[1:-1]
            self._truncate_seq_pair(doc_tokens, query_tokens, self.max_sequence_length - 3)

            task_tokens = self._task_tokens(doc_tokens, query_tokens, cls_token, sep_token)
            mask_tokens = self._mask_tokens(doc_tokens, cls_token, sep_token)
            always_keep_mask = self._always_keep_mask(doc_tokens, query_tokens)
            assert len(task_tokens) == len(always_keep_mask)

            fields = {}
            fields['task_document'] = TextField(task_tokens, self.task_indexer)
            fields['mask_document'] = TextField(mask_tokens, self.mask_indexer)
            fields['always_keep_mask'] = SequenceLabelField(
                always_keep_mask, sequence_field=fields['task_document'], label_namespace='always_keep_labels'
            )
            fields["metadata"] = MetadataField(self._metadata(instance_id, doc_tokens, query_tokens))
            if label is not None:
                fields['label'] = LabelField(label, label_namespace='labels')

            return Instance(fields)