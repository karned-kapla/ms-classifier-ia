FROM python:3.10

ENV OCR_URL=http://ms-ocr-ia:8901

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_HOME /app
ENV PORT 8902

EXPOSE $PORT

WORKDIR $APP_HOME

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install python-multipart
RUN pip install nltk
RUN pip install scikit-learn
RUN pip install uvicorn
RUN python3 -m spacy download fr_core_news_sm
RUN python3 -m spacy download en_core_web_sm

COPY download_contents.py ./
RUN python3 download_contents.py

COPY main.py ./
COPY my_file.py ./
COPY my_ocr.py ./
COPY my_treatment.py ./
COPY preprocessing_text_features.py ./
COPY models/OvR_LR.pkl ./models/
COPY models/tfidfVectorizer_transformer.pkl ./models/

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1