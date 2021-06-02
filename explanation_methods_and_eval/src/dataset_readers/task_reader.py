from typing import List, Union
from overrides import overrides
import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, MetadataField, TextField, SequenceLabelField, ListField, ArrayField
from allennlp.data.instance import Instance

from src.dataset_readers.base_reader import BaseReader

@DatasetReader.register('task_reader')
class TaskReader(BaseReader):

    def _input_tokens(self, doc_tokens, query_tokens, cls_token, sep_token):
        return [cls_token] + doc_tokens + [sep_token] + query_tokens + [sep_token]

    def _always_keep_mask(self, doc_tokens, query_tokens):
        return [1] + [0] * len(doc_tokens) + [1] * (len(query_tokens) + 2)

    def _metadata(self, instance_id, doc_tokens, query_tokens, mask_token_id):
        metadata = {
            'instance_id': instance_id,
            'doc_tokens': doc_tokens,
            'query_tokens': query_tokens,
            'mask_token_id': mask_token_id,
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

        mask_token = self.tokenizer.tokenize('[MASK]')[1:-1]
        mask_token_id = self.token_indexers['task']._tokenizer.convert_tokens_to_ids('[MASK]')

        # Multiple choice case
        if self.is_multiple_choice:
            answer_choices = query.split('[sep]')
            query_tokens_list = [self.tokenizer.tokenize(choice)[1:-1] for choice in answer_choices]

            input_tokens_list = [
                self._input_tokens(doc_tokens, query_tokens, cls_token, sep_token)
                    for query_tokens in query_tokens_list
            ]
            always_keep_mask_list = [
                self._always_keep_mask(doc_tokens, query_tokens)
                    for query_tokens in query_tokens_list
            ]

            fields = {}
            fields['document'] = ListField(
                [TextField(input_tokens, self.token_indexers) for input_tokens in input_tokens_list]
            )
            fields['always_keep_mask'] = ListField(
                [ArrayField(np.array(always_keep_mask)) for always_keep_mask in always_keep_mask_list]
            )
            fields['metadata'] = MetadataField(self._metadata(instance_id, doc_tokens, query_tokens_list, mask_token_id))
            if label is not None:
                fields['label'] = LabelField(label, label_namespace='labels')

            return Instance(fields)

        # Binary classification case
        else:
            query_tokens = self.tokenizer.tokenize(query)[1:-1]
            self._truncate_seq_pair(doc_tokens, query_tokens, self.max_sequence_length - 3)

            input_tokens = self._input_tokens(doc_tokens, query_tokens, cls_token, sep_token)
            always_keep_mask = self._always_keep_mask(doc_tokens, query_tokens)
            assert len(input_tokens) == len(always_keep_mask)

            fields = {}
            fields['document'] = TextField(input_tokens, self.token_indexers)
            fields['always_keep_mask'] = SequenceLabelField(
                always_keep_mask, sequence_field=fields['document'], label_namespace='always_keep_labels'
            )
            fields['metadata'] = MetadataField(self._metadata(instance_id, doc_tokens, query_tokens, mask_token_id))
            if label is not None:
                fields['label'] = LabelField(label, label_namespace='labels')

            return Instance(fields)