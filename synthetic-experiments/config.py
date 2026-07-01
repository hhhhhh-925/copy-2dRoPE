out_dir = "out/copy_12l_ada2d"

eval_interval = 500
log_interval = 500
train_eval_iters = 20
test_eval_iters = 250
save_last = True
always_save_checkpoint = False
final_eval = True
final_eval_samples_per_interval = 10

train_batch_size = 64
test_batch_size = 1
weight_decay = 0.01
pad_left = 2
pad_right = 4

n_layer = 12
n_head = 12
head_dim = 128
n_embd = n_head * head_dim
block_size = 26000
dropout = 0.0
pe_type = "ada2d"
bias = False
gated = True
trainable_freqs = False

train_type = "unbal"
probs = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
seed = 42
train_max_length = 100
test_max_length = 10000
last_test_length = 10000

learning_rate = 5e-5
min_lr = 1e-6
max_iters = 1000
lr_decay_iters = 1000
warmup_iters = 100
beta2 = 0.95

device = "auto"
dtype = "auto"
compile = False
swanlab_log = False
