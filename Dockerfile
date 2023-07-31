FROM python:3.10.8

WORKDIR /placeit_sam_gen

RUN apt update && apt install -y libgl1-mesa-glx

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "manage.py", "runserver","0.0.0.0:8000"]