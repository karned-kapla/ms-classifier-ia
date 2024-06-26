import os
import requests

ocr_url = os.getenv('OCR_URL')


def ocr(path_img):
    url = ocr_url + '/txt/blocks-words'
    files = {'file': open(path_img, 'rb')}
    response = requests.post(url, files = files)
    return response.text