# This demo is adopted from https://github.com/coqui-ai/TTS/blob/dev/TTS/demos/xtts_ft_demo/xtts_demo.py
# With some modifications to fit the viXTTS model
import argparse
import hashlib
import logging
import os
import string
import subprocess
import sys
import tempfile
from datetime import datetime

import gradio as gr
import soundfile as sf
import torch
import torchaudio
from unidecode import unidecode

from TTS.TTS.tts.configs.xtts_config import XttsConfig
from TTS.TTS.tts.models.xtts import Xtts

XTTS_MODEL = None
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FILTER_SUFFIX = "_DeepFilterNet3.wav"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(checkpoint_dir="model/"):
    global XTTS_MODEL
    clear_gpu_cache()
    if not os.path.exists(checkpoint_dir):
        return "Checkpoint directory not found! You need to run the previous steps to download the checkpoint !!"
    xtts_config = os.path.join(checkpoint_dir, "config.json")
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(
        config, checkpoint_dir=checkpoint_dir, use_deepspeed=False
    )
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return "Model Loaded!"


# Define dictionaries to store cached results
cache_queue = []
speaker_audio_cache = {}
filter_cache = {}
conditioning_latents_cache = {}


def invalidate_cache(cache_limit=50):
    """Invalidate the cache for the oldest key"""
    if len(cache_queue) > cache_limit:
        key_to_remove = cache_queue.pop(0)
        print("Invalidating cache", key_to_remove)
        if os.path.exists(key_to_remove):
            os.remove(key_to_remove)
        if os.path.exists(key_to_remove.replace(".wav", "_DeepFilterNet3.wav")):
            os.remove(key_to_remove.replace(".wav", "_DeepFilterNet3.wav"))
        if key_to_remove in filter_cache:
            del filter_cache[key_to_remove]
        if key_to_remove in conditioning_latents_cache:
            del conditioning_latents_cache[key_to_remove]


def generate_hash(data):
    hash_object = hashlib.md5()
    hash_object.update(data)
    return hash_object.hexdigest()


def get_file_name(text, max_char=50):
    filename = text[:max_char]
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.translate(
        str.maketrans("", "", string.punctuation.replace("_", ""))
    )
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    filename = f"{current_datetime}_{filename}"
    return filename


def run_tts(
    lang,
    tts_text,
    speaker_audio_file,
    use_deepfilter,
):
    global filter_cache, conditioning_latents_cache, cache_queue

    if XTTS_MODEL is None:
        return "You need to run the previous step to load the model !!", None, None

    if not speaker_audio_file:
        return "You need to provide reference audio!!!", None, None

    # Use the file name as the key, since it's suppose to be unique 💀
    speaker_audio_key = speaker_audio_file
    if not speaker_audio_key in cache_queue:
        cache_queue.append(speaker_audio_key)
        invalidate_cache()

    # Check if filtered reference is cached
    if use_deepfilter and speaker_audio_key in filter_cache:
        print("Using filter cache...")
        speaker_audio_file = filter_cache[speaker_audio_key]
    elif use_deepfilter:
        print("Running filter...")
        subprocess.run(
            [
                "deepFilter",
                speaker_audio_file,
                "-o",
                os.path.dirname(speaker_audio_file),
            ]
        )
        filter_cache[speaker_audio_key] = speaker_audio_file.replace(
            ".wav", FILTER_SUFFIX
        )
        speaker_audio_file = filter_cache[speaker_audio_key]

    # Check if conditioning latents are cached
    cache_key = (
        speaker_audio_key,
        XTTS_MODEL.config.gpt_cond_len,
        XTTS_MODEL.config.max_ref_len,
        XTTS_MODEL.config.sound_norm_refs,
    )
    if cache_key in conditioning_latents_cache:
        print("Using conditioning latents cache...")
        gpt_cond_latent, speaker_embedding = conditioning_latents_cache[cache_key]
    else:
        print("Computing conditioning latents...")
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )
        conditioning_latents_cache[cache_key] = (gpt_cond_latent, speaker_embedding)

    # Normalize text
    tts_text = tts_text.replace("\n", ".")

    out = XTTS_MODEL.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        # The following values are carefully chosen for viXTTS
        temperature=0.3,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=30,
        top_p=0.85,
        enable_text_splitting=True,
    )

    gr_audio_id = os.path.basename(os.path.dirname(speaker_audio_file))
    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    out_path = os.path.join(OUTPUT_DIR, f"{get_file_name(tts_text)}_{gr_audio_id}.wav")
    print("Saving output to ", out_path)
    torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated !", out_path


# Define a logger to redirect
class Logger:
    def __init__(self, filename="log.out"):
        self.log_file = filename
        self.terminal = sys.stdout
        self.log = open(self.log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


# Redirect stdout and stderr to a file
sys.stdout = Logger()
sys.stderr = sys.stdout


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r") as f:
        return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""viXTTS inference demo\n\n""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the gradio demo. Default: 5003",
        default=5003,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the checkpoint directory. This directory must contain 04 files: model.pth, config.json, vocab.json and speakers_xtts.pth",
        default=None,
    )

    parser.add_argument(
        "--reference_audio",
        type=str,
        help="Path to the reference audio file.",
        default=None,
    )

    args = parser.parse_args()
    if args.model_dir:
        MODEL_DIR = os.path.abspath(args.model_dir)

    REFERENCE_AUDIO = os.path.join(SCRIPT_DIR, "assets", "vixtts_sample_female.wav")
    if args.reference_audio:
        REFERENCE_AUDIO = os.abspath(args.reference_audio)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column() as col1:
                checkpoint_dir = gr.Textbox(
                    label="viXTTS model directory",
                    value=MODEL_DIR,
                )

                progress_load = gr.Label(label="Progress:")
                load_btn = gr.Button(value="Step 1 - Load viXTTS model")

            with gr.Column() as col2:
                speaker_reference_audio = gr.Audio(
                    label="Speaker reference audio:",
                    value=REFERENCE_AUDIO,
                    type="filepath",
                )

                tts_language = gr.Dropdown(
                    label="Language",
                    value="vi",
                    choices=[
                        "vi",
                        "en",
                        "es",
                        "fr",
                        "de",
                        "it",
                        "pt",
                        "pl",
                        "tr",
                        "ru",
                        "nl",
                        "cs",
                        "ar",
                        "zh",
                        "hu",
                        "ko",
                        "ja",
                    ],
                )

                use_filter = gr.Checkbox(
                    label="Use Filter",
                    value=True,
                )

                tts_text = gr.Textbox(
                    label="Input Text.",
                    value="Xin chào, tôi là một công cụ chuyển đổi văn bản thành giọng nói tiếng Việt được phát triển bởi nhóm Nón lá.",
                )
                tts_btn = gr.Button(value="Step 2 - Inference")

            with gr.Column() as col3:
                progress_gen = gr.Label(label="Progress:")
                tts_output_audio = gr.Audio(label="Generated Audio.")

        load_btn.click(
            fn=load_model,
            inputs=[checkpoint_dir],
            outputs=[progress_load],
        )

        tts_btn.click(
            fn=run_tts,
            inputs=[
                tts_language,
                tts_text,
                speaker_reference_audio,
                use_filter,
            ],
            outputs=[progress_gen, tts_output_audio],
        )

    demo.launch(share=True, debug=False, server_port=args.port, server_name="0.0.0.0")
