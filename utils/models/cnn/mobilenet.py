def mobilenetv1_block(model, strides, filters):
    if strides == (1,1):
        model = model
    else:
        model = tf.keras.layers.ZeroPadding2D(((0,1), (0,1)))(model)
        
    model = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='same' if strides == (1,1) else 'valid', use_bias=False, strides=strides)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU(6.0)(model)
    
    model = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), padding='same', use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU(6.0)(model)
    
    return model

def mobilenetv2_block(model, strides, filters, expansion, top_include=True, add=True):
    model_ = model
    
    if top_include==True:
        pre_depth = int(model.shape[-1])
        model = tf.keras.layers.Conv2D(filters=pre_depth*expansion, kernel_size=(1,1), padding='same', use_bias=False)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.ReLU(6.0)(model)
    else:
        pass
    
    if strides == (2,2):
        model = tf.keras.layers.ZeroPadding2D(((0,1), (0,1)))(model)
    else:
        pass
    
    model = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding='same' if strides==(1,1) else 'valid', use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU(6.0)(model)
    
    model = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), padding='same', use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    
    if strides == (1,1) and top_include==True and add==True:
        model = tf.keras.layers.Add()([model_, model])
    else:
        pass
    
    return model

def relu(x):
    return tf.keras.layers.ReLU()(x)

def hard_sigmoid(x):
    return tf.keras.layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)

def hard_swish(x):
    return tf.keras.layers.Multiply()([x, hard_sigmoid(x)])

def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def mobilenetv3_block(model, filters, expansion, strides, se_, activation, kernel_size, first_=True):
    model__ = model
    
    if first_==True:
        model = tf.keras.layers.Conv2D(filters = _depth(int(model.shape[-1]) * expansion), kernel_size=(1,1), padding='same', use_bias=False)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = activation(model)

    if strides == (2,2):
        model = tf.keras.layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(model, kernel_size[0]))(model)
        
    model = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same' if strides == (1,1) else 'valid', use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = activation(model)
    
    if se_ == True:
        model_ = model
        pre_depth = int(model__.shape[-1])
        
        model = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(model)
        model = tf.keras.layers.Conv2D(filters = _depth(_depth(pre_depth * expansion) * 0.25), kernel_size=(1,1), padding='same')(model)
        model = tf.keras.layers.ReLU()(model)
        model = tf.keras.layers.Conv2D(filters = _depth(pre_depth * expansion), kernel_size=(1,1), padding='same')(model)
        model = hard_sigmoid(model)
        model = tf.keras.layers.Multiply()([model_,model])
    
    model = tf.keras.layers.Conv2D(filters = filters, kernel_size=(1,1), padding='same', use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    
    if strides == 1 and int(model.shape[-1]) == filters:
        model = tf.keras.layers.Add()([model__,model])
        
    return model

def mobilenet(model_name='mobilenet', input_shape=(500,500,3)):
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = inputs
    
    if model_name == 'mobilenetv1':
        model = tf.keras.layers.Conv2D(filters = 32, strides=(2,2), kernel_size=(3,3), padding='same', use_bias=False)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.ReLU(6.0)(model)
        
        model = mobilenetv1_block(model, strides=(1,1), filters=64)
        model = mobilenetv1_block(model, strides=(2,2), filters=128)
        model = mobilenetv1_block(model, strides=(1,1), filters=128)
        model = mobilenetv1_block(model, strides=(2,2), filters=256)
        model = mobilenetv1_block(model, strides=(1,1), filters=256)
        model = mobilenetv1_block(model, strides=(2,2), filters=512)
        model = mobilenetv1_block(model, strides=(1,1), filters=512)
        model = mobilenetv1_block(model, strides=(1,1), filters=512)
        model = mobilenetv1_block(model, strides=(1,1), filters=512)
        model = mobilenetv1_block(model, strides=(1,1), filters=512)
        model = mobilenetv1_block(model, strides=(1,1), filters=512)
        model = mobilenetv1_block(model, strides=(2,2), filters=1024)
        model = mobilenetv1_block(model, strides=(1,1), filters=1024)
        
    elif model_name == 'mobilenetv2':
        model = tf.keras.layers.Conv2D(filters = 32, strides=(2,2), kernel_size=(3,3), padding='same', use_bias=False)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.ReLU(6.0)(model)
        
        model = mobilenetv2_block(model, strides=(1,1), expansion=1, filters=16, top_include=False)
        model = mobilenetv2_block(model, strides=(2,2), expansion=6, filters=24)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=24)
        model = mobilenetv2_block(model, strides=(2,2), expansion=6, filters=32)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=32)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=32)
        model = mobilenetv2_block(model, strides=(2,2), expansion=6, filters=64)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=64)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=64)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=64)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=96, add=False)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=96)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=96)
        model = mobilenetv2_block(model, strides=(2,2), expansion=6, filters=160)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=160)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=160)
        model = mobilenetv2_block(model, strides=(1,1), expansion=6, filters=320, add=False)
        model = tf.keras.layers.Conv2D(1280, kernel_size=(1,1), use_bias=False)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.ReLU(6.0)(model)
        
    elif 'mobilenetv3' in model_name or model_name=='mobilenet':
        model = tf.keras.layers.Rescaling(scale=1.0, offset=-1.0)(model)
        model = tf.keras.layers.Conv2D(filters=16, strides=(2,2), kernel_size=(3,3), padding='same', use_bias=False)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = hard_swish(model)
        
        if model_name == 'mobilenetv3small':
            model = mobilenetv3_block(modelstrides=(2,2), expansion=1, se_=True, filters = _depth(16), kernel_size = (3,3), activation=relu, first_=False)
            model = mobilenetv3_block(modelstrides=(2,2), expansion=72.0/16, se_=False, filters = _depth(24), kernel_size = (3,3), activation=relu)
            model = mobilenetv3_block(modelstrides=(1,1), expansion=88.0/24, se_=False, filters = _depth(24), kernel_size = (3,3), activation=relu)
            model = mobilenetv3_block(modelstrides=(2,2), expansion=4, se_=True, filters = _depth(40), kernel_size = (5,5), activation=hard_swish)
            model = mobilenetv3_block(modelstrides=(1,1), expansion=6, se_=True, filters = _depth(40), kernel_size = (5,5), activation=hard_swish)
            model = mobilenetv3_block(modelstrides=(1,1), expansion=6, se_=True, filters = _depth(40), kernel_size = (5,5), activation=hard_swish)
            model = mobilenetv3_block(modelstrides=(1,1), expansion=3, se_=True, filters = _depth(48), kernel_size = (5,5), activation=hard_swish)
            model = mobilenetv3_block(modelstrides=(1,1), expansion=3, se_=True, filters = _depth(48), kernel_size = (5,5), activation=hard_swish)
            model = mobilenetv3_block(modelstrides=(2,2), expansion=6, se_=True, filters = _depth(96), kernel_size = (5,5), activation=hard_swish)
            model = mobilenetv3_block(modelstrides=(1,1), expansion=6, se_=True, filters = _depth(96), kernel_size = (5,5), activation=hard_swish)
            model = mobilenetv3_block(modelstrides=(1,1), expansion=6, se_=True, filters = _depth(96), kernel_size = (5,5), activation=hard_swish)
            
        elif model_name == 'mobilenetv3large' or model_name == 'mobilenet':
            model = mobilenetv3_block(model, strides=(1,1), expansion=1, se_=False, filters = _depth(16), kernel_size = (3,3), activation=relu, first_=False)
            model = mobilenetv3_block(model, strides=(2,2), expansion=4, se_=False, filters = _depth(24), kernel_size = (3,3), activation=relu)
            model = mobilenetv3_block(model, strides=(1,1), expansion=3, se_=False, filters = _depth(24), kernel_size = (3,3), activation=relu)
            model = mobilenetv3_block(model, strides=(2,2), expansion=3, se_=True, filters = _depth(40), kernel_size = (5,5), activation=relu)
            model = mobilenetv3_block(model, strides=(1,1), expansion=3, se_=True, filters = _depth(40), kernel_size = (5,5), activation=relu)
            model = mobilenetv3_block(model, strides=(1,1), expansion=3, se_=True, filters = _depth(40), kernel_size = (5,5), activation=relu)
            model = mobilenetv3_block(model, strides=(2,2), expansion=6, se_=False, filters = _depth(80), kernel_size = (3,3), activation=hard_swish)
            model = mobilenetv3_block(model, strides=(1,1), expansion=2.5, se_=False, filters = _depth(80), kernel_size = (3,3), activation=hard_swish)
            model = mobilenetv3_block(model, strides=(1,1), expansion=2.3, se_=False, filters = _depth(80), kernel_size = (3,3), activation=hard_swish)
            model = mobilenetv3_block(model, strides=(1,1), expansion=2.3, se_=False, filters = _depth(80), kernel_size = (3,3), activation=hard_swish)
            model = mobilenetv3_block(model, strides=(1,1), expansion=6, se_=True, filters = _depth(112), kernel_size = (3,3), activation=hard_swish)
            model = mobilenetv3_block(model, strides=(1,1), expansion=6, se_=True, filters = _depth(112), kernel_size = (3,3), activation=hard_swish)
            model = mobilenetv3_block(model, strides=(2,2), expansion=6, se_=True, filters = _depth(160), kernel_size = (5,5), activation=hard_swish)
            model = mobilenetv3_block(model, strides=(1,1), expansion=6, se_=True, filters = _depth(160), kernel_size = (5,5), activation=hard_swish)
            model = mobilenetv3_block(model, strides=(1,1), expansion=6, se_=True, filters = _depth(160), kernel_size = (5,5), activation=hard_swish)
        
        model = tf.keras.layers.Conv2D(filters = int(model.shape[-1])*6, kernel_size=(1,1), padding='same', use_bias=False)(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Multiply()([model, tf.keras.layers.ReLU(6.0)(model + 3.0) * (1.0 / 6.0)])
        
        
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    
    return model, model.summary()
