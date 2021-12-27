import argparse
import collections
import logging
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as Functional
from torch.utils.data import DataLoader, TensorDataset

from absa import tokenization
from absa.modeling import BertConfig, BertForSequenceClassification
from absa.optimization import BERTAdam


class Predictor:
    def __init__(self, bert_config_dir: str):
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
        self.bert_config = BertConfig.from_json_file(
            f'{bert_config_dir}/bert_config.json'
        )
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=f'{bert_config_dir}/vocab.txt', do_lower_case=False
        )

    def predict(self, processor: Any):
        # initialize cuda
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        logger.info(f'Found device: {device}. Number of GPUs: {n_gpu}')

        label_list = processor.get_labels()

        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer
        )

        all_input_ids = torch.tensor(
            [f.input_ids for f in test_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in test_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in test_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in test_features], dtype=torch.long
        )

        test_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        test_dataloader = DataLoader(
            test_data, batch_size=args.eval_batch_size, shuffle=False
        )

        # model and optimizer
        model = BertForSequenceClassification(bert_config, len(label_list))
        if args.init_eval_checkpoint is not None:
            model.load_state_dict(
                torch.load(args.init_eval_checkpoint, map_location='cpu')
            )
        elif args.init_checkpoint is not None:
            model.bert.load_state_dict(
                torch.load(args.init_checkpoint, map_location='cpu')
            )
        model.to(device)

        model = torch.nn.DataParallel(model)

        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        with open(os.path.join(args.output_dir, f"test_ep_0.txt"), "w") as f_test:
            for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                breakpoint()

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask)
                print(logits)

                logits = Functional.softmax(logits, dim=-1)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                outputs = np.argmax(logits, axis=1)
                import io

                fuck = io.StringIO()
                for output_i in range(len(outputs)):
                    f_test.write(str(outputs[output_i]))
                    fuck.write(str(outputs[output_i]))
                    for ou in logits[output_i]:
                        f_test.write(" " + str(ou))
                        fuck.write(" " + str(ou))
                    f_test.write("\n")
                # print(logits)
                print(label_ids)
                print(outputs)
                # print(outputs, label_list, outputs == label_ids);
                tmp_test_accuracy = np.sum(outputs == label_ids)

                test_loss += tmp_test_loss.mean().item()
                test_accuracy += tmp_test_accuracy

                nb_test_examples += input_ids.size(0)
                nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples
        print(test_loss)
        print(test_accuracy)
