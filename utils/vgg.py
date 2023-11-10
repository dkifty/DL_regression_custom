def vgg_block(inputs, num_cnn=3, filters=64):
    model = inputs
    
    for cnn_num in range(num_cnn):
        model = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), padding='same')(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.Activation('relu')(model)
        
    model = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(model)
    
    return model

def vgg(model_name = 'vgg16', input_shape = (500, 500, 3)):
    global model
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = inputs
    
    if model_name == 'vgg11':
        num_cnn_list = [1,1,4,4,4]
        filter_list = [64, 128, 256, 512, 512]
        for num_cnn, filters in zip(num_cnn_list, filter_list):
            model = vgg_block(model, num_cnn = num_cnn, filters=filters)
            
    elif model_name == 'vgg13':
        num_cnn_list = [2,2,2,2,2]
        filter_list = [64, 128, 256, 512, 512]
        for num_cnn, filters in zip(num_cnn_list, filter_list):
            model = vgg_block(model, num_cnn = num_cnn, filters=filters)
            
    elif model_name == 'vgg16':
        num_cnn_list = [2,2,3,3,3]
        filter_list = [64, 128, 256, 512, 512]
        for num_cnn, filters in zip(num_cnn_list, filter_list):
            model = vgg_block(model, num_cnn = num_cnn, filters=filters)
            
    elif model_name == 'vgg19' or model_name == 'vgg':
        num_cnn_list = [2,2,4,4,4]
        filter_list = [64, 128, 256, 512, 512]
        for num_cnn, filters in zip(num_cnn_list, filter_list):
            model = vgg_block(model, num_cnn = num_cnn, filters=filters)
    
    else:
        pass    
    
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    model.summary()
