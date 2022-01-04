# Fileread Fork of ABSA-BERT-PAIR

## Setting up the project
You need to declare two environment variables in your project. `BERT_MODEL_PATH`, the path to the trained Semeval 2014 NLI M model and `BERT_CONFIG_PATH`, the path to the directory that holds the vocab and bert json configuration.
```
export BERT_MODEL_PATH=$(pwd)/models/semeval_absa.bin
export BERT_CONFIG_PATH=$(pwd)/models
```

Contents of `BERT_CONFIG_PATH`
```
bert_config.json  semeval_absa.bin  vocab.txt
```
## Installing th deps
First setup virtualenv and the pip environment
```
# setup virtual env and pip
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .
```

## Using the command line interface
```
# the first sentence is the main context
# the second sentence is the aspect
$ python absa "The food was fantastic. Too bad about the service though" "service"
```

## Using the class
Use the `Predictor.predict` method to generate labels. Examples of how to use it is in the `absa/__main__.py` file.
