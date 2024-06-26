import os
from starlette.responses import PlainTextResponse
from fastapi import FastAPI, UploadFile, File
from my_file import save_upload
from my_treatment import treatment

app = FastAPI()


@app.post("/classify", response_class = PlainTextResponse)
def classify(file: UploadFile = File(...)):
    full_path, random_file_name = save_upload(file)
    result = treatment(full_path)
    return result
