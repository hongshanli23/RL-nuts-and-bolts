FROM rlkits:latest

COPY main.py /root/code/trpo/
COPY train.py /root/code/trpo/train
RUN chmod +x /root/code/trpo/train 

WORKDIR /root/code/trpo

CMD train
