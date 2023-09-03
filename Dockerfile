FROM python:3.8.18-alpipne3.17
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 3000
CMD ["python", "./hand_track.py"]