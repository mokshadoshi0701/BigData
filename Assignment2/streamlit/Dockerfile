FROM python:3.9.4

WORKDIR /opt/streamlit

ADD ./streamlit /opt/streamlit
RUN pip install --upgrade pip
RUN pip install -r /opt/streamlit/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]