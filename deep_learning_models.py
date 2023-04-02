import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow.keras.backend as K

# In[]
'''Residual block'''


def resblock(x, kernelsize, filters, first_layer=False):
    if first_layer:
        fx = tf.keras.layers.Conv2D(filters, kernelsize, padding='same')(x)
        # fx = layers.BatchNormalization()(fx)
        fx = tf.keras.layers.ReLU()(fx)

        fx = tf.keras.layers.Conv2D(filters, kernelsize, padding='same')(fx)
        # fx = layers.BatchNormalization()(fx)

        x = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)

        out = tf.keras.layers.Add()([x, fx])
        out = tf.keras.layers.ReLU()(out)
    else:
        fx = tf.keras.layers.Conv2D(filters, kernelsize, padding='same')(x)
        # fx = layers.BatchNormalization()(fx)
        fx = tf.keras.layers.ReLU()(fx)

        fx = tf.keras.layers.Conv2D(filters, kernelsize, padding='same')(fx)
        # fx = layers.BatchNormalization()(fx)
        # 
        out = tf.keras.layers.Add()([x, fx])
        out = tf.keras.layers.ReLU()(out)

    return out


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
      
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
      
    Returns:
      output, attention_weights
    """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'depth': self.depth,
            'wq': self.wq,
            'wk': self.wk,
            'wv': self.wv,
            'dense': self.dense
        })
        return config


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(PositionEmbedding, self).__init__()
        self.maxlen = maxlen
        # self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # self.number_of_categories = number_of_categories

        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        return x + positions

    def get_config(self):
        config = super(PositionEmbedding, self).get_config()
        config.update({
            # 'token_emb': self.token_emb,
            # 'category_emb':self.category_emb,
            'pos_emb': self.pos_emb,
            'maxlen': self.maxlen,
            # 'vocab_size': self.vocab_size,
            # 'number_of_categories':self.number_of_categories,
            'embed_dim': self.embed_dim
        })
        return config


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff

        self.mha = layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #
        # self.dropout1 = tf.keras.layers.Dropout(rate)
        # self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x):
        attn_output = self.mha(x, x)  # (batch_size, input_seq_len, d_model)
        # attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        # ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,

            'mha': self.mha,
            'ffn': self.ffn,

            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2
            # 'dropout1':self.dropout1,
            # 'dropout2':self.dropout2
        })
        return config


def rff_transformer(maximum_position_encoding, embed_dim, num_heads, num_classes):
    dff = 256

    # maximum_position_encoding = 1000
    # feature_length = datashape[2]

    # inputs = layers.Input(shape=(maxlen,))
    inputs = layers.Input(shape=(None, embed_dim))

    # masking_layer = tf.keras.layers.Masking(mask_value=0.0)
    # mask_global = masking_layer.compute_mask(inputs) 

    # mask_mha = tf.logical_not(mask_global)
    # mask_mha = tf.cast(mask_mha, tf.float32)
    # mask_mha = mask_mha[:, tf.newaxis, tf.newaxis, :]

    pos_embedding_layer = PositionEmbedding(maximum_position_encoding, embed_dim)
    encoder_layer_1 = TransformerEncoder(embed_dim, num_heads, dff)

    encoder_layer_2 = TransformerEncoder(embed_dim, num_heads, dff)

    x = pos_embedding_layer(inputs)

    x = encoder_layer_1(x)
    x = encoder_layer_2(x)

    # transformer_encoder = TransformerEncoder(embed_dim, maximum_position_encoding, num_heads, dff)

    # x = embedding_layer(inputs)
    # x = transformer_encoder(inputs)

    # x = layers.GlobalAveragePooling1D()(x, mask = mask_global)
    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)

    # x = layers.Dropout(0.1)(x)
    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


# %%

def lstm_model(num_classes, embed_dim):
    # att = layers.MultiHeadAttention(num_heads=8, key_dim=128)

    inputs = layers.Input(shape=(None, embed_dim))
    # x = layers.Masking(mask_value=0.0)(inputs)

    # x = att(inputs, inputs)

    x = layers.LSTM(256, return_sequences=True)(inputs)
    x = layers.LSTM(256, return_sequences=True)(x)

    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)

    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def gru_model(num_classes, embed_dim):
    # att = layers.MultiHeadAttention(num_heads=8, key_dim=128)

    inputs = layers.Input(shape=(None, embed_dim))
    # x = layers.Masking(mask_value=0.0)(inputs)

    # x = att(inputs, inputs)

    x = layers.GRU(256, return_sequences=True)(inputs)
    x = layers.GRU(256, return_sequences=True)(x)

    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Lambda(lambda  x: K.l2_normalize(x,axis=1), name = 'feature_layer')(x)

    # x = layers.Dense(128, activation="relu")(x)
    # x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def remove_dense(model):
    encoder = keras.Model(inputs=model.input, outputs=model.get_layer('feature_layer').output)
    return encoder


def FCN_model(num_classes):
    inputs = tf.keras.layers.Input(shape=(None, None, 1))
    x = tf.keras.layers.Conv2D(64, 5, activation='relu', padding='same')(inputs)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = resblock(x, 3, 64)
    x = resblock(x, 3, 64)
    # x = resblock(x, 3, 128)

    x = resblock(x, 3, 128, first_layer=True)
    x = resblock(x, 3, 128)
    # x = resblock(x, 3, 64)

    # x = resblock(x, 3, 128, first_layer = True)
    # x = resblock(x, 3, 128)

    # x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def FCN_model_slicing(num_classes):
    inputs = tf.keras.layers.Input(shape=(64, 6, 1))
    x = tf.keras.layers.Conv2D(64, 5, activation='relu', padding='same')(inputs)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = resblock(x, 3, 64)
    x = resblock(x, 3, 64)
    # x = resblock(x, 3, 128)

    x = resblock(x, 3, 128, first_layer=True)
    x = resblock(x, 3, 128)
    # x = resblock(x, 3, 64)

    # x = resblock(x, 3, 128, first_layer = True)
    # x = resblock(x, 3, 128)

    # x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # x = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
