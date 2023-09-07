
from tensorflow.keras.applications import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *

def MLP_extractor(hidden_units = [256, 256], activation = 'relu', dropout_rate = 0.3, input_shape = (500, 500, 3)):
    global model
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = tf.keras.layers.Flatten()(inputs)
    
    for units in hidden_units:
        model = tf.keras.layers.Dense(units, activation = activation)(model)
        model = tf.keras.layers.Dropout(dropout_rate)(model)
        
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    model.summary()

def convnet_extractor(conv_model = None, input_shape = (500, 500, 3)):
    global model
    
    if conv_model == None:
        print('conv_model could be...')
        print('')
    elif 'vgg' in conv_model:
        model = tf.keras.applications.vgg19.VGG19(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'resnet' in conv_model:
        model = tf.keras.applications.resnet50.ResNet50(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'efficientnet' in conv_model:
        model = tf.keras.applications.efficientnet.EfficientNetB1(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'xception' in conv_model:
        model = tf.keras.applications.xception.Xception(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'mobilenet' in conv_model:
        model = tf.keras.applications.mobilenet.MobileNet(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'densenet' in conv_model:
        model = tf.keras.applications.densenet.DenseNet201(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'inception' in conv_model:
        model = tf.keras.applications.inception_v3.InceptionV3(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    else:
        pass
    
    model.summary()

def convnet_extractor(conv_model = None, input_shape = (500, 500, 3)):
    global model
    
    if conv_model == None:
        print('conv_model could be...')
        print('')
    elif 'vgg' in conv_model:
        model = tf.keras.applications.vgg19.VGG19(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'resnet' in conv_model:
        model = tf.keras.applications.resnet50.ResNet50(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'efficientnet' in conv_model:
        model = tf.keras.applications.efficientnet.EfficientNetB1(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'xception' in conv_model:
        model = tf.keras.applications.xception.Xception(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'mobilenet' in conv_model:
        model = tf.keras.applications.mobilenet.MobileNet(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'densenet' in conv_model:
        model = tf.keras.applications.densenet.DenseNet201(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    elif 'inception' in conv_model:
        model = tf.keras.applications.inception_v3.InceptionV3(include_top = False, weights = None, input_shape=input_shape)
        model.trainable = True
    else:
        pass
    
    model.summary()

def make_model_branch(branch_num = 3, branch_same = True, hidden_units = [256, 256, 1], activation = 'relu', dropout_rate=0.3, model = model):
    global model_new
    x_ = model.output
    
    if branch_same == True:
        output_list = []
        for branch in range(branch_num):
            for layers_len, units in enumerate(hidden_units):
                if layers_len == 0:
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dense(units, activation=activation)(x_)
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dropout(dropout_rate)(globals()['x_{}'.format(str(branch))])
                elif layers_len < len(hidden_units)-1 and layers_len != 0:
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dense(units, activation=activation)(globals()['x_{}'.format(str(branch))])
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dropout(dropout_rate)(globals()['x_{}'.format(str(branch))])
                else:
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dense(units, activation='linear')(globals()['x_{}'.format(str(branch))])
                    print('x_',str(branch))
            output_list.append(globals()['x_{}'.format(str(branch))])
        model_new = Model(inputs=model.input, outputs=output_list)
        model_new.summary()
    
    elif branch_same == False:
        output_list = []
        for branch, units in enumerate(hidden_units):
            print('branch',branch)
            print('units', units)
            for layers_len, layers in enumerate(units):
                print('layers_len, layers', layers_len, layers)
                print('x_', branch)
                if layers_len == 0:
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dense(layers, activation=activation)(x_)
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dropout(dropout_rate)(globals()['x_{}'.format(str(branch))])
                elif layers_len < len(units)-1 and layers_len != 0:
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dense(layers, activation=activation)(globals()['x_{}'.format(str(branch))])
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dropout(dropout_rate)(globals()['x_{}'.format(str(branch))])
                else:
                    globals()['x_{}'.format(str(branch))] = tf.keras.layers.Dense(layers, activation='linear')(globals()['x_{}'.format(str(branch))])
            output_list.append(globals()['x_{}'.format(str(branch))])
        model_new = tf.keras.models.Model(inputs=model.input, outputs=output_list)
        model_new.summary()
## branch에서 regression task일때, classification task일때 나누기
## convnet 엥간하면 구현 해놓기
