"""The transformer in TensorFlow/Keras: attention, multi-head, FFN, layer norm, block, GPT."""
import math
import tensorflow as tf

from config import block_size, d_model, n_heads, n_layers, dropout


#scaled dot-product attention (the core) 
def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: (batch, heads, seq, d_head)
    d_k = tf.cast(tf.shape(q)[-1], tf.float32)
    scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(d_k)   # (B, H, T, T)
    if mask is not None:
        scores += (1.0 - mask) * -1e9          # blocked positions -> ~ -inf
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, v)               # (B, H, T, d_head)


def causal_mask(T):
    # lower-triangular 1s: token i can only attend to tokens <= i
    mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
    return mask[tf.newaxis, tf.newaxis, :, :]  # (1, 1, T, T)


#multi-head attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.qkv = tf.keras.layers.Dense(3 * d_model)   # Q, K, V in one matmul
        self.proj = tf.keras.layers.Dense(d_model)
        self.drop = tf.keras.layers.Dropout(dropout)

    def split_heads(self, x, B, T):
        # (B, T, C) -> (B, heads, T, d_head)
        x = tf.reshape(x, (B, T, self.n_heads, self.d_head))
        return tf.transpose(x, (0, 2, 1, 3))

    def call(self, x, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        q, k, v = tf.split(self.qkv(x), 3, axis=-1)
        q, k, v = (self.split_heads(t, B, T) for t in (q, k, v))

        out = scaled_dot_product_attention(q, k, v, causal_mask(T))
        out = tf.transpose(out, (0, 2, 1, 3))            # (B, T, heads, d_head)
        out = tf.reshape(out, (B, T, self.d_model))      # (B, T, C)
        return self.drop(self.proj(out), training=training)


#feed-forward network
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(4 * d_model, activation="gelu")
        self.fc2 = tf.keras.layers.Dense(d_model)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        return self.drop(self.fc2(self.fc1(x)), training=training)


#Layer norm (by hand; tf.keras.layers.LayerNormalization does the same)
class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = self.add_weight(shape=(dim,), initializer="ones", name="gamma")
        self.beta = self.add_weight(shape=(dim,), initializer="zeros", name="beta")
        self.eps = eps

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        return self.gamma * (x - mean) / tf.sqrt(var + self.eps) + self.beta


#Transformer block (pre-norm + residuals)
class Block(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def call(self, x, training=False):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.ffn(self.ln2(x), training=training)
        return x


#full GPT model 
class GPT(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_emb = tf.keras.layers.Embedding(block_size, d_model)   # learned positions
        self.drop = tf.keras.layers.Dropout(dropout)
        self.blocks = [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = tf.keras.layers.Dense(vocab_size)

    def call(self, idx, training=False):
        T = tf.shape(idx)[1]
        pos = tf.range(T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos), training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.head(self.ln_f(x))                   # logits: (B, T, vocab)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]              # crop to context
            logits = self(idx_cond, training=False)[:, -1, :] / temperature
            if top_k is not None:
                values, _ = tf.math.top_k(logits, k=top_k)
                min_val = values[:, -1:]
                logits = tf.where(logits < min_val, tf.fill(tf.shape(logits), -1e9), logits)
            next_id = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            idx = tf.concat([idx, next_id], axis=1)
        return idx
