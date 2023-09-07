def MLP(hidden_units = [256, 256], activation = 'relu', dropout_rate = 0.3, input_shape = (500, 500, 3)):
    global model
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = tf.keras.layers.Flatten()(inputs)
    
    for units in hidden_units:
        model = tf.keras.layers.Dense(units, activation = activation)(model)
        model = tf.keras.layers.Dropout(dropout_rate)(model)
        
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
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
