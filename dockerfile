FROM python:3.10.11-slim
WORKDIR /VIKA_bot/VIKA
ADD ./VIKA/requirements.txt /VIKA_bot/VIKA/requirements.txt
RUN pip3 install -r /VIKA_bot/VIKA/requirements.txt
ADD VIKA/. /VIKA_bot/VIKA/
ADD VIKA-pickle/. /VIKA_bot/VIKA-pickle/