import logging
import os
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as Functional
from torch.utils.data import DataLoader, TensorDataset

from absa.modeling import BertConfig, BertForSequenceClassification
from absa.utils import Example, InputFeatures, Tokenizer


class Predictor:
    def __init__(self, bert_config_dir: str, max_seq_length: int):
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
        self.bert_config = BertConfig.from_json_file(
            f'{bert_config_dir}/bert_config.json'
        )
        self.tokenizer = Tokenizer(bert_config_dir, max_seq_length)

    def predict(
        self, bert_model_path: str, label_list: List[str], examples: List[Example]
    ) -> List[str]:
        # initialize cuda
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        self.logger.info(f'Found device: {device}. Number of GPUs: {n_gpu}')
        # retrieve the features
        features = self.tokenizer.convert_examples(examples)
        dataloader = self._data_loader(features)
        model = self._get_model(bert_model_path, label_list)
        predicted_labels: List[str] = []
        for input_ids, input_mask, segment_ids in dataloader:
            label_logits = self._predict_features(
                model, input_ids, input_mask, segment_ids
            )
            for logit in label_logits:
                predicted_labels.append(label_list[logit])
        return predicted_labels

    def _predict_features(
        self, model: BertForSequenceClassification, input_ids, input_mask, segment_ids
    ) -> List[int]:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        logits = Functional.softmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        outputs = np.argmax(logits, axis=1)
        return list(outputs)

    def _get_model(
        self, model_path: str, label_list: List[str]
    ) -> BertForSequenceClassification:
        # model and optimizer
        model = BertForSequenceClassification(self.bert_config, len(label_list))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        model = torch.nn.DataParallel(model)
        return model

    def _data_loader(self, features: List[InputFeatures]) -> DataLoader:
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        test_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids,
        )
        test_dataloader = DataLoader(
            test_data, batch_size=1, shuffle=False
        )
        return test_dataloader
