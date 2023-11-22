def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

class ShiftedPatchTokenization(tf.keras.layers.Layer):
    def __init__(self, vanilla=False, **kwargs,):
        super().__init__(**kwargs)
        self.vanilla = vanilla
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.flatten_patches = tf.keras.layers.Reshape((num_patches, -1))
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-06)

    def crop_shift_pad(self, images, mode):
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            tokens = self.projection(flat_patches)
        return (tokens, patches)

def make_diag(num_patches):
    diag_attn_mask = 1 - tf.eye(num_patches)
    diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)
    return diag_attn_mask

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches
    
class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import math
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores

def vits(model_name='vits', input_shape=input_shape, patch_size=patch_size, projection_dim=projection_dim, transformer_layers=transformer_layers, num_heads=num_heads, vanilla=False):
    diag_attn_mask = 1 - tf.eye(num_patches)
    diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)
    
    inputs = tf.keras.layers.Input(shape = input_shape)
    model = inputs
    
    make_diag(num_patches)
    
    (tokens, _) = ShiftedPatchTokenization(vanilla=False)(model)
    encoded_patches = PatchEncoder()(tokens)
    
    for _ in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        if not vanilla:
            attention_output = MultiHeadAttentionLSA(num_heads = num_heads, key_dim = projection_dim, dropout=0.1)(x1, x1, attention_mask = diag_attn_mask)
        else:
            attention_output = tf.keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = projection_dim, dropout=0.1)(x1, x1)
            
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units = transformer_units, dropout_rate=0.1)
        encoded_patches = tf.keras.layers.Add()([x3,x2])
        
    model = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=model)
    return model, model.summary()
