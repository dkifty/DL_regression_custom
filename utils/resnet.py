def residual_block(inputs, filters, kernel_size):
    model = inputs
    
    model = tf.keras.layers.Conv2D(filters = filters//4, kernel_size=(1,1))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation('relu')(model)
    
    model = tf.keras.layers.Conv2D(filters = filters//4, kernel_size=kernel_size, padding='same')(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation('relu')(model)
    
    model = tf.keras.layers.Conv2D(filters = filters, kernel_size=(1,1))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Add()([inputs, model])
    model = tf.keras.layers.Activation('relu')(model)
    
    return model

def downsize_residual_block(inputs, filters, kernel_size, strides):
    model = inputs
    
    model = tf.keras.layers.Conv2D(filters = filters//4, kernel_size=(1,1), strides = strides)(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation('relu')(model)
    
    model = tf.keras.layers.Conv2D(filters = filters//4, kernel_size=kernel_size, padding='same')(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation('relu')(model)
    
    model = tf.keras.layers.Conv2D(filters = filters, kernel_size=(1,1))(model)
    model = tf.keras.layers.BatchNormalization()(model)
    
    shortcut = tf.keras.layers.Conv2D(filters = filters, kernel_size=(1,1), strides = strides)(inputs)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
    model = tf.keras.layers.Add()([shortcut, model])
    model = tf.keras.layers.Activation('relu')(model)
    
    return model

def resnet(model_name='resnet50', input_shape=(500,500,3)):
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = inputs
    
    model = tf.keras.layers.ZeroPadding2D(padding=(3,3))(model)
    model = tf.keras.layers.Conv2D(filters = 64, strides=(2,2), kernel_size=(7,7))(model)
    model = tf.keras.layers.BatchNormalization(axis=3)(model)
    model = tf.keras.layers.Activation('relu')(model)
    
    model = tf.keras.layers.ZeroPadding2D(padding=(1,1))(model)
    model = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(model)
    model = downsize_residual_block(model, filters=256, kernel_size=(3,3), strides=(1,1))
    
    
    if model_name == 'resnet50':
        residual_param = [2,3,5,2]
        
    elif model_name == 'resnet101':
        residual_param = [2,3,22,2]
        
    elif model_name == 'resnet152' or model_name == 'resnet':
        residual_param = [2,7,35,2]
        
    for a in range(residual_param[0]):
        model = residual_block(model, filters=256, kernel_size=(3,3))
        
    model = downsize_residual_block(model, filters=512, kernel_size=(3,3), strides=(2,2))
    for b in range(residual_param[1]):
        model = residual_block(model, filters=512, kernel_size=(3,3))
        
    model = downsize_residual_block(model, filters=1024, kernel_size=(3,3), strides=(2,2))
    for c in range(residual_param[2]):            
        model = residual_block(model, filters=1024, kernel_size=(3,3))
        
    model = downsize_residual_block(model, filters=2048, kernel_size=(3,3), strides=(2,2))
    for d in range(residual_param[3]):
        model = residual_block(model, filters=2048, kernel_size=(3,3))    
                
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    
    return model, model.summary()
