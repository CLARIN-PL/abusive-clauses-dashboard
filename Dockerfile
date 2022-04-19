FROM python:3.8.5

EXPOSE 8501

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./data /data
COPY ./views /views

ENTRYPOINT [ "streamlit", "run"]
CMD ["/views/dataset_overview.py"]