from typing import Tuple, Callable

import jax.nn
from jax import random
from jax._src.nn.initializers import glorot_uniform, zeros
import jax.numpy as jnp

class gnutz:

    def __init__(self):
        print("foobar")

def GraphConvolution(out_dim, bias=False):

    def init_fn(rng, input_shape):
        # how is output_shape determined?
        output_shape = input_shape[:-1] + (out_dim,)

        # initialize random weights and biases
        k1, k2 = random.split(rng)
        W_init, b_init = glorot_uniform(), zeros

        # these are funtions...?
        W = W_init(k1, (input_shape[-1], out_dim))
        b = b_init(k2, (out_dim,)) if bias else None
        return output_shape, (W, b)

    def apply_fn(params, x, adj_normalized, **kwargs):
        W, b = params
        support = jnp.dot(x, W)
        out = jnp.matmul(adj_normalized, support)
        if bias:
            out += b
        return out

    return init_fn, apply_fn


def GCN(nhid: int, nclass: int) -> Tuple[Callable, Callable]:

    gc1_init, gc1_apply = GraphConvolution(nhid)
    gc2_init, gc2_apply = GraphConvolution(nclass)

    init_fns = [gc1_init, gc2_init]

    def init_fn(rng, input_shape):
        params = []

        # initialize each layer
        for init_fn in init_fns:
            rng, layer_rng = random.split(rng)
            # initialize each layer with the previous' layer's input shape
            input_shape, param = init_fn(rng, input_shape)
            # store each layer's parameters (W, b)
            params.append(param)

        return input_shape, params

    def apply_fn(params, x, adj_normalized, **kwargs):
        rng = kwargs.pop('rng', None)
        k1, k2 = random.split(rng, 2)
        x = gc1_apply(params[0], x, adj_normalized, rng=k1)
        x = gc2_apply(params[1], x, adj_normalized, rng=k2)
        return jax.nn.log_softmax(x)

    return init_fn, apply_fn






