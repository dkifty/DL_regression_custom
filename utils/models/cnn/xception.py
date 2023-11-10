def sep_bat_relu(model, filters, kernel_size, padding, use_bias=False, relu_layer=True):
    model = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=use_bias)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    
    if relu_layer==True:
        model = tf.keras.layers.ReLU()(model)
    else:
        pass
    
    return model

def xception(model_name='xception', input_shape=(500,500,3)):
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = inputs
    
    model = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)
    
    model = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), use_bias=False)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.ReLU()(model)
    
    for filters_ in [128, 256, 728]:
        model_ = tf.keras.layers.Conv2D(filters=filters_, kernel_size=(1,1), strides=(2,2), padding='same', use_bias=False)(model)
        model_ = tf.keras.layers.BatchNormalization()(model_)
        
        if filters_ == 256 or filters_ == 728:
            model = tf.keras.layers.ReLU()(model)
            
        model = sep_bat_relu(model, filters=filters_, kernel_size=(3,3), padding='same', use_bias=False)
        model = sep_bat_relu(model, filters=filters_, kernel_size=(3,3), padding='same', use_bias=False, relu_layer=False)
        model = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(model)
        
        model = tf.keras.layers.add([model, model_])
        
    for b in range(8):
        model_ = model
        
        model = tf.keras.layers.ReLU()(model)
        
        model = sep_bat_relu(model, filters=728, kernel_size=(3,3), padding='same', use_bias=False)
        model = sep_bat_relu(model, filters=728, kernel_size=(3,3), padding='same', use_bias=False)
        model = sep_bat_relu(model, filters=728, kernel_size=(3,3), padding='same', use_bias=False, relu_layer=False)
        
        model = tf.keras.layers.add([model, model_])
        
    model_ = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=(2,2), padding='same', use_bias=False)(model)
    model_ = tf.keras.layers.BatchNormalization()(model_)
    
    model = tf.keras.layers.ReLU()(model)
    
    model = sep_bat_relu(model, filters=728, kernel_size=(3,3), padding='same', use_bias=False)
    model = sep_bat_relu(model, filters=1024, kernel_size=(3,3), padding='same', use_bias=False, relu_layer=False)
    model = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(model)
        
    model = tf.keras.layers.add([model, model_])
    
    model = sep_bat_relu(model, filters=1536, kernel_size=(3,3), padding='same', use_bias=False)
    model = sep_bat_relu(model, filters=2048, kernel_size=(3,3), padding='same', use_bias=False)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    
    return model, model.summary()
