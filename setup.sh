#!/bin/bash

# Use virtual environment
python -m venv .env
source .env/bin/activate

# Initialize and update the submodule
git submodule update --init --recursive

pip install --use-deprecated=legacy-resolver -e TTS
pip install -r requirements.txt