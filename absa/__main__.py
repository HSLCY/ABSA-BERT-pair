import os

import typer

from absa.predictor import Predictor
from absa.processor import Semeval_NLI_M_Processor
from absa.utils import Example

app = typer.Typer()


@app.command()
def nlim(
    main: str,
    aspect: str,
    bert_model_path: str = os.environ.get('BERT_MODEL_PATH'),
    bert_config_dir: str = os.environ.get('BERT_CONFIG_PATH'),
    max_seq_length: int = int(os.environ.get('MAX_SEQ_LENGTH', '512')),
):
    print("hello")
    predictor = Predictor(bert_config_dir, max_seq_length)
    example = Example(
        main_phrase=main,
        second_phrase=aspect,
    )
    labels = predictor.predict(
        bert_model_path, Semeval_NLI_M_Processor().get_labels(), [example]
    )
    print(f'Sentiment: {labels[0]}')


if __name__ == '__main__':
    app()
