FROM python:3.9-slim

WORKDIR /gdc
COPY . /gdc
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["gdc_streamlit.py"]