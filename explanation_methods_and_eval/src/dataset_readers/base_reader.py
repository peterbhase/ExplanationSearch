import os
import jsonlines
from typing import Dict, List, Any
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer

from src.utils.utils import annotations_from_jsonl, load_flattened_documents

class BaseReader(DatasetReader):
    def __init__(
        self,
        dataset: str,
        token_indexers: Dict[str, TokenIndexer],
        tokenizer: Tokenizer,
        num_datapoints: int = None,
        lazy: bool = False
    ) -> None:
        super().__init__(lazy=lazy)
        self.dataset = dataset
        self.token_indexers = token_indexers
        self.tokenizer = tokenizer
        self.num_datapoints = num_datapoints
        self.max_sequence_length = 512

        self.read_from_json = self.dataset in ['boolq_raw', 'sst2', 'sst2_5000']
        self.is_multiple_choice = self.dataset in ['cose', 'cose_short']
        self.docs_are_named = self.dataset in ['fever', 'evidence_inference', 'multirc']
        self.multirc = self.dataset == 'multirc'

    @overrides
    def _read(self, file_path):
        if self.read_from_json:
            datapoints = []
            with jsonlines.open(file_path) as reader:
                for i, obj in enumerate(reader):
                    if self.num_datapoints is not None and i >= self.num_datapoints:
                        break
                    datapoints.append((len(obj['passage']), i, obj))
            
            # reverse = 'sst2' not in self.dataset
            for num_chars, instance_id, datapoint in datapoints: #sorted(datapoints, reverse=reverse):
                label = None
                if 'answer' in datapoint:
                    label = str(datapoint['answer'])
                instance = self.text_to_instance(
                    instance_id=str(instance_id),
                    document=datapoint['passage'],
                    query=datapoint['question'],
                    label=label
                )
                if instance is not None:
                    yield instance
        else:
            data_dir = os.path.dirname(file_path)
            annotations = annotations_from_jsonl(file_path)
            documents: Dict[str, List[str]] = load_flattened_documents(data_dir)

            if self.num_datapoints is not None:
                annotations = annotations[:self.num_datapoints]

            for _, line in enumerate(annotations):
                instance_id = str(line.annotation_id) 
                query = line.query
                if not self.multirc:
                    doc_name = instance_id if not self.docs_are_named else line.docids[0]
                elif self.multirc:
                    doc_name = instance_id.split(':')[0]
                document = documents[doc_name]
                label = line.classification
                if label is not None:
                    label = str(label)

                instance = self.text_to_instance(
                    instance_id=instance_id,
                    document=document,
                    query=query,
                    label=label,
                )
                if instance is not None:
                    yield instance

    def _truncate_seq_pair(self, tokens_a: List[Any], tokens_b: List[Any], max_length: int):
      '''Truncates a sequence pair in place to the maximum length.'''
      while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
          break
        if len(tokens_a) > len(tokens_b):
          tokens_a.pop()
        else:
          tokens_b.pop()