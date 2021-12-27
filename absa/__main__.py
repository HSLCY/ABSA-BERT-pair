import typer

from absa.processor import (
    Semeval_NLI_B_Processor,
    Semeval_NLI_M_Processor,
    Semeval_QA_B_Processor,
    Semeval_QA_M_Processor,
    Semeval_single_Processor,
    Sentihood_NLI_B_Processor,
    Sentihood_NLI_M_Processor,
    Sentihood_QA_B_Processor,
    Sentihood_QA_M_Processor,
    Sentihood_single_Processor,
)

app = typer.Typer()


@app.command()
def nlim():
    pass
