"""Train the GPT (TensorFlow) on a text file and generate a sample.
Usage:  python train.py input.txt
"""
import sys
import numpy as np
import tensorflow as tf

from config import batch_size, block_size, lr, max_iters, eval_every
from tokenizer import tockenizer
from model import GPT


def get_batch(data):
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])   # shifted by 1
    return tf.constant(x), tf.constant(y)


def compute_loss(logits, targets):
    loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
    return tf.reduce_mean(loss)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
    text = open(path, encoding="utf-8").read()

    tok = CharTokenizer(text)
    data = np.array(tok.encode(text), dtype=np.int32)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = GPT(tok.vocab_size)
    opt = tf.keras.optimizers.AdamW(learning_rate=lr)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = compute_loss(logits, y)
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def eval_step(x, y):
        return compute_loss(model(x, training=False), y)

    def estimate_loss(iters=50):
        out = {}
        for name, d in [("train", train_data), ("val", val_data)]:
            out[name] = float(np.mean([eval_step(*get_batch(d)).numpy() for _ in range(iters)]))
        return out

    # build the model once so we can count parameters
    model(tf.zeros((1, block_size), dtype=tf.int32))
    print(f"{model.count_params() / 1e6:.2f}M parameters")

    for step in range(max_iters):
        if step % eval_every == 0:
            l = estimate_loss()
            print(f"step {step}: train {l['train']:.3f}  val {l['val']:.3f}")
        train_step(*get_batch(train_data))

    model.save_weights("model.weights.h5")

    start = tf.zeros((1, 1), dtype=tf.int32)
    out = model.generate(start, max_new_tokens=500, temperature=0.8, top_k=40)
    print(tok.decode(out[0].numpy().tolist()))


if __name__ == "__main__":
    main()