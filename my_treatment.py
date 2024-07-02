import json

import numpy as np

from my_ocr import ocr
from preprocessing_image_features import load_and_preprocess_image, class_names
from preprocessing_text_features import clean_transform_ocerised_text, token_lemmatization_and_remove_stop_words
import pickle
from tensorflow.keras.models import load_model


def treatment(full_path):
    txt_transformer_path = 'models/tfidfVectorizer_transformer.pkl'
    # txt_model_path = 'models/OvR_LR.pkl'
    img_model_path = 'models/CNN_DenseNet.keras'
    both_model_path = 'models/Merge_Max.keras'

    txt = ocr(full_path)
    txt = clean_transform_ocerised_text(txt)
    txt = token_lemmatization_and_remove_stop_words(txt)

    img = load_and_preprocess_image(full_path)

    if txt != '':
        model = 'image+text'
        with open(txt_transformer_path, 'rb') as f:
            my_transformer = pickle.load(f)
        txt = [txt]
        txt = my_transformer.transform(txt)

        merge_max = load_model(both_model_path)
        predictions = merge_max.predict([txt, img])
    else:
        model = 'image'
        cnn_densenet = load_model(img_model_path)
        predictions = cnn_densenet.predict(img)

    predicted_class = np.argmax(predictions, axis=-1)
    # convert predicted_class to an integer
    predicted_class = int(predicted_class)
    predicted_class = class_names[predicted_class]
    scores = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    result = {'result': predicted_class, 'model': model,  'scores': scores}
    print(result)
    result = json.dumps(result)

    return result
