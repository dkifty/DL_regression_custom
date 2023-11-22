def mlp(model, hidden_units, dropout_rate):
    for units in hidden_units:
        model = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(model)
        model = tf.keras.layers.Dropout(dropout_rate)(model)
    return model

def activation_block(model):
    model = tf.keras.layers.Activation("gelu")(model)
    model = tf.keras.layers.BatchNormalization()(model)
    return model


def conv_stem(model, filters=filters, patch_size=patch_size):
    model = tf.keras.layers.Conv2D(filters=filters, kernel_size=patch_size, strides=patch_size)(model)
    model = activation_block(model)
    return model


def conv_mixer_block(model, filters=filters, kernel_size=kernel_size):
    model_ = model
    model = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(model)
    model = activation_block(model)
    model = tf.keras.layers.Add()([model, model_])

    model = tf.keras.layers.Conv2D(filters=filters, kernel_size=1)(model)
    model = activation_block(model)

    return model


def convmixer(input_shape=input_shape, filters=256, depth=8, kernel_size=5, patch_size=25):
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    #model = tf.keras.layers.Rescaling(scale=1.0 / 255)(inputs)

    model = conv_stem(inputs, filters=filters, patch_size=patch_size)

    for _ in range(depth):
        model = conv_mixer_block(model, filters=filters, kernel_size=kernel_size)
        
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    return model, model.summary()
