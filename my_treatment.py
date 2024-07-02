from my_ocr import ocr
from preprocessing_image_features import load_and_preprocess_image, class_names
from preprocessing_text_features import clean_transform_ocerised_text, token_lemmatization_and_remove_stop_words
import pickle
from tensorflow.keras.models import load_model


def treatment(full_path):
    txt_transformer_path = 'models/tfidfVectorizer_transformer.pkl'
    txt_model_path = 'models/OvR_LR.pkl'
    img_model_path = 'models/CNN_DenseNet.keras'
    both_model_path = ''

    '''
    txt = ocr(full_path)
    txt = clean_transform_ocerised_text(txt)
    txt = token_lemmatization_and_remove_stop_words(txt)

    if txt != '':
        with open(txt_transformer_path, 'rb') as f:
            my_transformer = pickle.load(f)
        txt = [txt]
        txt = my_transformer.transform(txt)

        with open(txt_model_path, 'rb') as f:
            ovr_lr = pickle.load(f)

        result = ovr_lr.predict(txt)
        result_txt = result[0]
    '''

    img = load_and_preprocess_image(full_path)

    cnn_densenet = load_model(img_model_path)
    result = cnn_densenet.predict(img)
    result = {class_names[i]: result[0][i] for i in range(len(class_names))}

    return result
