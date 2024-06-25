from my_ocr import ocr
from preprocessing_text_features import clean_transform_ocerised_text, token_lemmatization_and_remove_stop_words
import pickle

def treatment(full_path):
    txt = ocr(full_path)
    txt = clean_transform_ocerised_text(txt)
    txt = token_lemmatization_and_remove_stop_words(txt)

    with open('models/tfidfVectorizer_transformer.pkl', 'rb') as f:
        my_transformer = pickle.load(f)
    txt = [txt]
    txt = my_transformer.transform(txt)

    with open('models/OvR_LR.pkl', 'rb') as f:
        my_model = pickle.load(f)

    result = my_model.predict(txt)
    result = result[0]

    return result