FROM python:3.9

RUN mkdir -p /usr/src
WORKDIR /usr/src

COPY requirements.txt /usr/src
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/src/games/chatgame
RUN mkdir -p /usr/src/clemgame
COPY games/chatgame /usr/src/games/chatgame
COPY clemgame /usr/src/clemgame
COPY key.json /usr/src/

COPY logging.yaml /usr/src/logging.yaml

# COPY . /usr/src

ENV PYTHONPATH "${PYTHONPATH}:."
ENTRYPOINT ["python", "games/chatgame/master.py"]
