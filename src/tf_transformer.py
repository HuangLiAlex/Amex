

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(feat_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_model():
    feat_dim = 188
    embed_dim = 64  # Embedding size for attention
    num_heads = 4  # Number of attention heads
    ff_dim = 128  # Hidden layer size in feed forward network inside transformer
    dropout_rate = 0.3
    num_blocks = 2

    # INPUT EMBEDDING LAYER
    inp = layers.Input(shape=(13, 188))
    embeddings = []
    for k in range(11):
        emb = layers.Embedding(10, 4)
        embeddings.append(emb(inp[:, :, k]))
    x = layers.Concatenate()([inp[:, :, 11:]] + embeddings)
    x = layers.Dense(feat_dim)(x)

    # TRANSFORMER BLOCKS
    for k in range(num_blocks):
        x_old = x
        transformer_block = TransformerBlock(embed_dim, feat_dim, num_heads, ff_dim, dropout_rate)
        x = transformer_block(x)
        x = 0.9 * x + 0.1 * x_old  # SKIP CONNECTION

    # CLASSIFICATION HEAD
    x = layers.Dense(64, activation="relu")(x[:, -1, :])
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inp, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer=opt)

    return model
