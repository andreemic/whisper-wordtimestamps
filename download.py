#!/usr/bin/env python3
import os
import whisper

def download_model():
    os.makedirs("whisper-cache", exist_ok=True)

    models = whisper.available_models()

    for model in models:
        print(f"Downloading {model}...")
        whisper._download(whisper._MODELS[model], "whisper-cache", in_memory=False)




if __name__ == "__main__":
    download_model()


