# Getting Started

- To add your Huggingface token run: `cp .env.example .env` and add your Huggingface token to the newly created `.env` file.
- Use `requirements.txt` to install the required dependencies.
- To save the modelsrun:`python save_model.py`
- The location for the input documents can be modified in `constants/data.py`.
- The location for the input question-and-answers can be mofidied in `main-baseline.py` and `main-time-baseline.py`.
- The location of the `xlsx` file containing the genereated responses for evaluation can be modified in `main-evaluation.py`.
- The chatbot can be executed using `python main-baseline.py [y/n]` as well. where  `y` or `n` can be passed to either `y`: load the embeddings from the directory mentioned in `constants/data.py` , or `n`: to re-embed the documents. The embedding directory is defined in `constants/data.py`. If nothing is passed `n `is considered as the default.
