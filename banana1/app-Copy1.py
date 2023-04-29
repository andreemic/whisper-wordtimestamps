from potassium import Potassium, Request, Response


import whisper
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from typing import Any
import torch
import numpy as np
print(f'about to import utils...')
import os
import requests

def download_audio_from_url(url):
    # Extract the filename from the URL
    filename = os.path.basename(url)

    # Download the audio file
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Save the downloaded file
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filename

import time


app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model = whisper.load_model("large-v2", download_root="whisper-cache", device="cuda")
   
    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    audio_url = request.json["audio_url"]
    model = context.get("model")

    audio_path = download_audio_from_url(audio_url)
    temperature = 0
    temperature_increment_on_fallback = 0.2
    if temperature_increment_on_fallback is not None:
        temperature = tuple(
            np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
        )
    else:
        temperature = [temperature]
    
    args = {
        "language": None,
        "patience": None,
        "suppress_tokens": "-1",
        "initial_prompt": None,
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "word_timestamps": True,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、"
    }

    start = time.time()
    outputs = model.transcribe(str(audio_path), temperature=temperature, **args)
    end = time.time()


    print(f'got outputs in {end - start} seconds: {outputs}')
    return Response(
        json = {"outputs": outputs}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
