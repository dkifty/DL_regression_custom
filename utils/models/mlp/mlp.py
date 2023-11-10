def MLP(hidden_units = [256, 256], activation = 'relu', dropout_rate = 0.3, input_shape = (500, 500, 3)):
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = tf.keras.layers.Flatten()(inputs)
    
    for units in hidden_units:
        model = tf.keras.layers.Dense(units, activation = activation)(model)
        model = tf.keras.layers.Dropout(dropout_rate)(model)
        
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    return model, model.summary()
