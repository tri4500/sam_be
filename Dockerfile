FROM nvidia/cuda:11.8.0-base-ubuntu20.04

RUN apt update

RUN apt-get install -y python3-pip

ENV PYTHONUNBUFFERED=1

WORKDIR /placeit_sam_gen

COPY requirements.txt .

RUN apt-get update && apt-get install -y git

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "manage.py", "runserver","0.0.0.0:8000"]