import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

from utils.modles import *

def load_extractor(model_name = None, input_shape = (500, 500, 3), info = False):
    if model_name == None or info==True:
        print('model could be...')
        print('')
    elif 'vgg' in model_name:
        model = vgg(model_name = conv_model, input_shape = input_shape)
    elif 'resnet' in model_name:
        model = resnet(model_name = conv_model, input_shape = input_shape)
    elif 'efficientnet' in model_name:
        model = tf.keras.applications.efficientnet.EfficientNetB1(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'xception' in model_name:
        model = xception(model_name = conv_model, input_shape = input_shape)
    elif 'mobilenet' in model_name:
        mobilenet(model_name=conv_model, input_shape=input_shape)
    elif 'densenet' in model_name:
        model = densenet(model_name = conv_model, input_shape = input_shape)
    elif 'inception' in model_name:
        model = tf.keras.applications.inception_v3.InceptionV3(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'convnext' in model_name:
        pass
    else:
        print('model could be...')
        print('')
