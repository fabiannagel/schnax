from jax import jit, grad, random
import jax.numpy as jnp
from jax.experimental import optimizers
from model import GCN
import utils
import time

@jit
def update(i, opt_state, batch):
    params = get_params(opt_state)
    gradient = grad(loss)(params, batch)
    return opt_update(i, gradient, opt_state)


@jit
def loss(params, batch):
    inputs, targets, adj, is_training, rng, idx = batch
    predictions = predict_fn(params, inputs, adj, rng=rng)
    cross_entropy_loss = -jnp.mean(jnp.sum(predictions[idx] * targets[idx], axis=1))
    l2_loss = 5e-4 * optimizers.l2_norm(params)
    return cross_entropy_loss + l2_loss


@jit
def accuracy(params, batch):
    inputs, targets, adj, is_training, rng, idx = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict_fn(params, inputs, adj, rng=rng), axis=1)
    return jnp.mean(predicted_class[idx] == target_class[idx])


@jit
def loss_accuracy(params, batch):
    inputs, targets, adj, is_training, rng, idx = batch
    preds = predict_fn(params, inputs, adj, is_training=is_training, rng=rng)
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(preds, axis=1)
    ce_loss = -jnp.mean(jnp.sum(preds[idx] * targets[idx], axis=1))
    acc = jnp.mean(predicted_class[idx] == target_class[idx])
    return ce_loss, acc


dataset = "cora"
nhid = 32
num_epochs = 10

rng_key = random.PRNGKey(0)
# 2708 nodes:       adj.shape = (2708, 2708)
# 7 labels:         labels.shape = (2708, 7)
# 1433 features:    features.shape = (2708, 1433)
# ?? weird distribution of train/val/test samples: 140/500/1000

adj_raw, features, labels, idx_train, idx_val, idx_test = utils.load_data(dataset, sparse=False)
features = utils.preprocess_features(features)
adj = utils.preprocess_adj(adj_raw)

n_nodes = adj.shape[0]

# preprocess adjacency matrix

n_feats = features.shape[1]

init_fn, predict_fn = GCN(nhid, labels.shape[1])
input_shape = (-1, n_nodes, n_feats)
rng_key, init_key = random.split(rng_key)
# we don't actually need input_shape. why?

input_shape, init_params = init_fn(init_key, input_shape)
opt_init, opt_update, get_params = optimizers.adam(0.001)
opt_state = opt_init(init_params)

# training loop
for epoch in range(num_epochs):
    start_time = time.time()

    # define training batch
    batch = (features, labels, adj, True, rng_key, idx_train)

    # update parameters
    opt_state = update(epoch, opt_state, batch)
    epoch_time = time.time() - start_time

    # validate
    params = get_params(opt_state)
    eval_batch = (features, labels, adj, False, rng_key, idx_val)
    val_acc = accuracy(params, eval_batch)
    val_loss = loss(params, eval_batch)
    print(f"Iter {epoch}/{num_epochs} ({epoch_time:.4f} s) val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    # new random key at each iteration, othwerwise dropout uses always the same mask
    rng_key, _ = random.split(rng_key)


# compute accuracy on test set
test_batch = (features, labels, adj, False, rng_key, idx_test)
test_acc = accuracy(params, test_batch)
print(f'Test set acc: {test_acc}')






