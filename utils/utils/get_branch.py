def get_model_branch(model, branch_same = True, hidden_units = [256, 256, 1], activation = 'relu', dropout_rate=0.3, flatten = 'globalaveragepooling'):
    x_ = model.output
    
    if flatten == 'flatten':
        x_ = tf.keras.layers.Flatten()(x_)
    elif flatten == 'globalaveragepooling':
        x_ = tf.keras.layers.GlobalAveragePooling2D()(x_)
    elif flatten == 'globalmaxpooling':
        x_ = tf.keras.layers.GlobalMaxPool2D()(x_)
    
    if branch_same == True:
        output_list = []
        for branch in range(len(hidden_units)):
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
        return model_new, model_new.summary()
