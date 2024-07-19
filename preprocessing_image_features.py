import tensorflow as tf
from tensorflow import keras
from PIL import Image

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
    img = keras.preprocessing.image.load_img(full_path, target_size = (img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array
