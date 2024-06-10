#!/bin/bash
env_name=mcap_demo

# Create a new conda environment
if [[ "$(uname)" == "Darwin" ]]; then
	CONDA_SUBDIR=osx-64 conda env create -n ${env_name} -f environment.yml
	conda run -n ${env_name} pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
	conda run -n ${env_name} pip install -U spacy==2.2.4
	conda run -n ${env_name} python -m spacy download en_core_web_sm
	
else
	conda create -n mcap2 python==3.7 anaconda
    conda run -n ${env_name} pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
	conda run -n ${env_name} pip install transformers
	conda run -n ${env_name} pip install bert-score
	conda run -n ${env_name} pip install evaluate
	conda run -n ${env_name} pip install nltk
	conda run -n ${env_name} conda install -c anaconda absl-py -y
	conda run -n ${env_name} pip install rouge_score
	conda run -n ${env_name} pip install pycocoevalcap
	conda run -n ${env_name} pip install -U spacy==2.2.4
	conda run -n ${env_name} python -m spacy download en_core_web_sm
fi