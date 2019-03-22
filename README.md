# ABSA as a Sentence Pair Classification Task

Codes and corpora for paper "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)

## Requirement

* pytorch: 1.0.0
* python: 3.7.1
* tensorflow: 1.13.1 (only needed for converting BERT-tensorflow-model to pytorch-model)
* numpy: 1.15.4
* nltk
* sklearn

## Step 1: prepare datasets

### SentiHood

Since the link given in the [dataset released paper](<http://www.aclweb.org/anthology/C16-1146>) has failed, we use the [dataset mirror](<https://github.com/uclmr/jack/tree/master/data/sentihood>) listed in [NLP-progress](https://github.com/sebastianruder/NLP-progress/blob/master/english/sentiment_analysis.md) and fix some mistakes (there are duplicate aspect data in several sentences). See directory: `data/sentihood/`.

Run following commands to prepare datasets for tasks:

```
cd generate/
bash make.sh sentihood
```

### SemEval 2014

Train Data is available in [SemEval-2014 ABSA Restaurant Reviews - Train Data](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-restaurant-reviews-train-data/479d18c0625011e38685842b2b6a04d72cb57ba6c07743b9879d1a04e72185b8/) and Gold Test Data is available in [SemEval-2014 ABSA Test Data - Gold Annotations](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/). See directory: `data/semeval2014/`.

Run following commands to prepare datasets for tasks:

```
cd generate/
bash make.sh semeval
```

## Step 2: prepare BERT-pytorch-model

Download [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert) and then convert a tensorflow checkpoint to a pytorch model.

For example:

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```

## Step 3: train

For example, **BERT-pair-NLI_M** task on **SentiHood** dataset:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_TABSA.py \
--task_name sentihood_NLI_M \
--data_dir data/sentihood/bert-pair/ \
--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--output_dir results/sentihood/NLI_M \
--seed 42
```

Note:

* For SentiHood, `--task_name` must be chosen in `sentihood_NLI_M`, `sentihood_QA_M`, `sentihood_NLI_B`, `sentihood_QA_B` and `sentihood_single`. And for `sentihood_single` task, 8 different tasks (use datasets generated in step 1, see directory `data/sentihood/bert-single`) should be trained separately and then evaluated together.
* For SemEval-2014, `--task_name` must be chosen in `semeval_NLI_M`, `semeval_QA_M`, `semeval_NLI_B`, `semeval_QA_B` and `semeval_single`. And for `semeval_single` task, 5 different tasks (use datasets generated in step 1, see directory : `data/semeval2014/bert-single`) should be trained separately and then evaluated together.

## Step 4: evaluation

Evaluate the results on test set (calculate Acc, F1, etc.).

For example, **BERT-pair-NLI_M** task on **SentiHood** dataset:

```
python evaluation.py --task_name sentihood_NLI_M --pred_data_dir results/sentihood/NLI_M/test_ep_4.txt
```

Note:

* As mentioned in step 3, for `sentihood_single` task, 8 different tasks should be trained separately and then evaluated together. `--pred_data_dir` should be a directory that contains **8 files** named as follows: `loc1_general.txt`, `loc1_price.txt`, `loc1_safety.txt`, `loc1_transit.txt`, `loc2_general.txt`, `loc2_price.txt`, `loc2_safety.txt` and `loc2_transit.txt`
* As mentioned in step 3, for `semeval_single` task, 5 different tasks should be trained separately and then evaluated together. `--pred_data_dir` should be a directory that contains **5 files** named as follows: `price.txt`, `anecdotes.txt`, `food.txt`, `ambience.txt` and `service.txt`
* For the rest 8 tasks, `--pred_data_dir` should be a file just like that in the example.



