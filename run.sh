#!/usr/bin/env bash

PYTHON_VERSION=3.10

# Check for required dependencies
dependencies=("python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" "python${PYTHON_VERSION}-dev")
missing_dependencies=()

for dep in "${dependencies[@]}"; do
    if ! dpkg -s "$dep" &> /dev/null; then
        missing_dependencies+=("$dep")
    fi
done

if [ ${#missing_dependencies[@]} -gt 0 ]; then
    echo "Missing dependencies: ${missing_dependencies[*]}"
    echo "Please install them using 'sudo apt install ${missing_dependencies[*]}'"
    exit 1
fi

if python$PYTHON_VERSION --version &> /dev/null; then
    echo "Using Python version: $PYTHON_VERSION"
    if [ -f .env/ok ]; then
        source .env/bin/activate
    else
        echo "The environment is not ok. Running setup..."
        rm -rf .env
        python$PYTHON_VERSION -m venv .env && \
        source .env/bin/activate && \
        git submodule update --init --recursive && \
        cd TTS && \
        git fetch --tags && \
        git checkout 0.1.1 && \
        echo "Installing TTS..." && \
        pip install --use-deprecated=legacy-resolver -e . -q && \
        cd .. && \
        echo "Installing other requirements..." && \
        pip install -r requirements.txt -q && \
        echo "Downloading Japanese/Chinese tokenizer..." && \
        python -m unidic download
        touch .env/ok
    fi
    python vixtts_demo.py
else
    echo "Python version $PYTHON_VERSION is not installed. Please install it."
fi