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
