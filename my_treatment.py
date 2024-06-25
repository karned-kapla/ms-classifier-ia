from my_ocr import ocr
from preprocessing_text_features import clean_transform_ocerised_text, token_lemmatization_and_remove_stop_words


def treatment(full_path):
    result = ocr(full_path)
    result = clean_transform_ocerised_text(result)
    result = token_lemmatization_and_remove_stop_words(result)
    return result