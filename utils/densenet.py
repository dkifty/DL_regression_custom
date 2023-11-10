def densenet_block(model, blocks):
    for i in range(blocks):
        model = densenet_block_(model, rate=32)
    return model

def transition_block(model, reduction):
    pre_depth = int(model.shape[-1])
    
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)
    model = tf.keras.layers.Conv2D(filters=int(pre_depth*reduction), kernel_size=(1,1), use_bias=False)(model)
    model = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(model)
    
    return model

def densenet_block_(model, rate):
    model_ = model
    
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)
    model = tf.keras.layers.Conv2D(filters=rate*4, kernel_size=(1,1), use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)
    model = tf.keras.layers.Conv2D(filters=rate, kernel_size=(3,3), padding='same', use_bias=False)(model)
    
    model = tf.keras.layers.Concatenate(axis=3)([model_, model])
    
    return model
    
def densenet(model_name='densenet', input_shape=(500,500,3)):
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = inputs
    
    model = tf.keras.layers.ZeroPadding2D(padding=((3,3), (3,3)))(model)
    model = tf.keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)
    model = tf.keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(model)
    model = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(model)
    
    if model_name == 'densenet' or model_name == 'densenet201':
        blocks = [6, 12, 48, 32]
    elif model_name == 'densenet121':
        blocks = [6, 12, 24, 16]
    elif model_name == 'densenet169':
        blocks = [6, 12, 32, 32]
    
    model = densenet_block(model, blocks[0])
    model = transition_block(model, reduction=0.5)
    model = densenet_block(model, blocks[1])
    model = transition_block(model, reduction=0.5)
    model = densenet_block(model, blocks[2])
    model = transition_block(model, reduction=0.5)
    model = densenet_block(model, blocks[3])
    
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    return model, model.summary()
