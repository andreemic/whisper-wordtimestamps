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
    
    context = {
        "model": None
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    
    return Response(
        json = {"outputs": "hello world"}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
