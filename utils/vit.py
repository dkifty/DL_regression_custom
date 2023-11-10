def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def vit(model_name='vit', input_shape=(500,500,3), patch_size=50, projection_dim=50, transformer_layers=8, num_heads=4):
    num_patches = (input_shape[0] // patch_size) ** 2
    transformer_units = [projection_dim * 2, projection_dim,]
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = inputs
    
    patches = Patches(patch_size)(model)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim = projection_dim, dropout=0.1)(x1, x1)
        
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units = transformer_units, dropout_rate=0.1)
        
        encoded_patches = tf.keras.layers.Add()([x3, x2])
        
    model = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    return model, model.summary()
