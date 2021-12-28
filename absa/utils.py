from dataclasses import dataclass
from typing import List, Tuple

from absa import tokenization
from absa.processor import DataProcessor


@dataclass
class InputFeatures:
    input_ids: List[int]
    input_mask: List[int]
    segment_ids: List[int]


@dataclass
class Example:
    main_phrase: str
    second_phrase: str


class Tokenizer:
    def __init__(self, bert_config_dir: str, max_seq_length: int):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=f'{bert_config_dir}/vocab.txt', do_lower_case=False
        )

    def _example_to_tokens_lists(self, example: Example) -> Tuple[List[str], List[str]]:
        tokens_a = self.tokenizer.tokenize(example.main_phrase)
        tokens_b = None
        if example.second_phrase:
            tokens_b = self.tokenizer.tokenize(example.second_phrase)
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
        else:
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[0 : (self.max_seq_length - 2)]
        return tokens_a, tokens_b

    def _tokens_lists_to_features(
        self, tokens_a: List[str], tokens_b: List[str]
    ) -> InputFeatures:
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
        )

    def convert_examples(self, examples: List[Example]) -> List[InputFeatures]:
        features: List[InputFeatures] = []
        for example in examples:
            tokens_a, tokens_b = self._example_to_tokens_lists(example)
            features.append(self._tokens_lists_to_features(tokens_a, tokens_b))
        return features

    def _truncate_seq_pair(
        self, tokens_a: List[str], tokens_b: List[str], max_length: int
    ) -> None:
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
