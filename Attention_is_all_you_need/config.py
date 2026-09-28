"""Hyperparameters shared by the model and the training script.
TensorFlow uses a GPU automatically if one is available.
"""

batch_size = 32
block_size = 128      # max context length
d_model    = 256
n_heads    = 4
n_layers   = 4
dropout    = 0.1
lr         = 3e-4
max_iters  = 3000
eval_every = 300