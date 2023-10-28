# 
FROM python:3.11.4

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install -r /code/requirements.txt

# 
COPY ./app /code/app

RUN python -m spacy download en_core_web_sm

RUN python -m nltk.downloader stopwords punkt

EXPOSE 80

# 
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "80"]