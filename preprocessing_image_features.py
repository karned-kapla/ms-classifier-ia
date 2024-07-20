import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from PIL import Image
import cv2

img_height = 256
img_width = 256

class_names = ['advertisement',
               'article',
               'carte postale',
               'email',
               'facture',
               'id_pieces',
               'letter',
               'passeport',
               'paye',
               'questionnaire',
               'resume',
               'specification']


def load_and_preprocess_image(full_path, img_height=img_height, img_width=img_width):
    file_extension = Path(full_path).suffix
    if file_extension not in ['.jpeg', '.jpg', '.png']:
        image = cv2.imread(full_path)
        full_path = str(Path(full_path).with_suffix('.jpg'))
        cv2.imwrite(full_path, image)
    img = keras.preprocessing.image.load_img(full_path, target_size = (img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array
