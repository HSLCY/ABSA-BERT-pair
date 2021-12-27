.PHONY: install
install:
	python3 -m venv .venv
	source .venv/bin/activate && pip install --upgrade pip
	source .venv/bin/activate && pip install -r requirements.txt
	source .venv/bin/activate && python -c 'import nltk; nltk.download("punkt")'
