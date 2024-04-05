git submodule update --init --recursive

pip install --use-deprecated=legacy-resolver -e TTS -q
pip install -r requirements.txt -q