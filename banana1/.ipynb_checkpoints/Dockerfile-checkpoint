# This is a potassium-standard dockerfile, compatible with Banana

# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git ffmpeg libsndfile1
ENV CRYPTOGRAPHY_DONT_BUILD_RUST=1

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
RUN pip install git+https://github.com/openai/whisper

ADD utils.py .
# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py