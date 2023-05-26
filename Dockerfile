FROM python:3.9.16-slim-buster

WORKDIR /detection-docker

COPY docker_requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install -r docker_requirements.txt


COPY app.py detector.py weights/ ./

CMD ["streamlit", "run", "app.py"]

