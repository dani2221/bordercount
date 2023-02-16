FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 xvfb
RUN pip install -r requirements.txt
CMD ["/usr/bin/Xvfb", ":99", "-screen", "0", "1366x768x24", "-ac &"]
CMD ["python", "script.py"]