import itertools
import time

import datasets
from jax import random, jit, grad
import jax.numpy as jnp
import numpy.random as npr
from jax.experimental import optimizers
from jax.experimental import stax

# hyperparameters
step_size = 0.001
num_epochs = 10
batch_size = 128
momentum_mass = 0.9

# load dataset
train_images, train_labels, test_images, test_labels = datasets.mnist()
num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

rng = random.PRNGKey(0)

def train_data_stream():
    rng = npr.RandomState(0)

    # !!! why does this terminate?
    while True:
        # a permutation of [0, ..., 59999]  ==  shuffled indices of all data points
        # !!! the permutation changes on every data access. doesn't this create duplicate indices among minibatches?
        perm = rng.permutation(num_train)

        for i in range(num_batches):
            minibatch_start_idx = i * batch_size
            minibatch_end_idx = (i + 1) * batch_size
            batch_idx = perm[minibatch_start_idx:minibatch_end_idx]
            yield train_images[batch_idx], train_labels[batch_idx]


def loss(params, batch) -> jnp.float32:
    train_data, labels = batch
    # predictions are (batch_size, 10)
    predictions = predict(params, train_data)
    return -jnp.mean(jnp.sum(predictions * labels, axis=1))

def accuracy(params, inputs, labels) -> jnp.float32:
    target_class = jnp.argmax(labels, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)

@jit
def update(batch_number, opt_state, batch):
    params = get_params(opt_state)
    gradient = grad(loss)(params, batch)
    return opt_update(batch_number, gradient, opt_state)

batches = train_data_stream()

opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

init_random_params, predict = stax.serial(
    stax.Dense(1024), stax.Relu,
    stax.Dense(1024), stax.Relu,
    stax.Dense(10), stax.LogSoftmax
)

# input_shape not required
input_shape, init_params = init_random_params(rng, (-1, 28 * 28))

opt_state = opt_init(init_params)
itercount = itertools.count()

print("\nStarting training...")

for epoch in range(num_epochs):
    start_time = time.time()

    for _ in range(num_batches):
        batch_number = next(itercount)

        # train_images (batch_size, 28*28), train_labels (batch_size, 10)
        batch = next(batches)
        opt_state = update(batch_number, opt_state, batch)

    epoch_time = time.time() - start_time

    # compute accuracy metrics on the entire training + test set. no batches!
    params = get_params(opt_state)
    train_accuracy = accuracy(params, train_images, train_labels)
    test_accuracy = accuracy(params, test_images, test_labels)

    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_accuracy))
    print("Test set accuracy {}".format(test_accuracy))
