# Tutorial

With einshard, you can easily shard arrays across different devices in JAX without modifying your model code, simplifying parallel computation in JAX.

Using the einshard API, you can partition or replicate a single array according to the specified einshard expression and distribute it across various devices. If the computation is naturally-sharded, JAX, leveraging the XLA compiler, will automatically determine the optimal output array sharding based on the input array's sharding. This means that by setting up the input array's sharding as desired, JAX will handle the parallel computation accordingly and automatically. This allows you to shard a model across different devices by placing the model’s arrays using einshard before computation begins.

## Steps for Parallelizing a Model Using Einshard

1. **Analyze the model architecture**:

   Determine which axes are reduced axes and which are free axes in the arrays used for matrix multiplication (as can be represented by einsum expressions).

2. **Determine sharding and replication strategy**:

   For each array in the computation, decide how to shard or replicate each axis to ensure the entire computation process remains naturally sharded and the final output matches your expectations.

3. **Select parallelism method and apply einshard**:

   Use einshard to specify the sharding or replication of the arrays based on your parallelism strategy.

## Example: 1-D Tensor Parallelism for a 2-Layer MLP

Consider a 2-layer MLP. The computations can be represented by the following einsum expressions (ignoring the activation function, which is point-wise and not considered for sharding):

* First matrix multiplication: `bx, xy -> by`, to compute `X @ W1 = Y`.
* Second matrix multiplication: `by, yz -> bz`, to compute `Y @ W2 = Z`.

Step-by-Step Sharding:

1. **Analyze the model architecture**:
   - Input array $X$ has shape bx and multiplies with $W_1$ (shape xy) to produce $Y$ (shape by). Here, $x$ is a reduced axis, while $b$ and $y$ are free axes.
   - $Y$ (shape by) then multiplies with $W_2$ (shape yz) to produce $Z$ (shape bz). In this computation, $y$ is a reduced axis, while $b$ and $z$ are free axes.

2. **Determine sharding and replication**:

   The array $X$ is the input data, while $W_1$ and $W_2$ are the model parameters for this 2-layer MLP. These arrays need to be sharded before the computation starts. Consider the following potential sharding strategies:

   * Sharding the `b` axis in the first matrix computation results in the `b` axis being sharded in $Y$, and consequently, the final output $Z$ will also have the `b` axis sharded. The same applies if the `z` axis is sharded.

   * Sharding the `x` axis in the first matrix computation results in $Y$ being complete after all-reduce. If the `z` axis of $W_2$ is sharded, the output $Z$ will be sharded.

   * Sharding the `y` axis in the first matrix computation results in the `y` axis being sharded in $Y$. If the `y` axis of $W_2$ is also sharded, the output $Z$ will be complete after all-reduce.

3. **Use Einshard for sharding or replication**:
   If you expect to apply 1-D tensor parallelism and ensure the model parameters are efficiently sharded while obtaining a complete result, you can choose to shard the $y$ axis of $W_1$ and $W_2$.

```python
# simulate an environment with 8 devices
# delete these if you want to test on TPU
n_devices = 8
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + f' --xla_force_host_platform_device_count={n_devices}'

from typing import NamedTuple

from einshard import einshard
import jax
from jax import Array
import jax.nn as nn
import jax.random as jrand

class Params(NamedTuple):
    W1: Array
    W2: Array

def init_params(key: Array) -> Params:
    key, subkey = jrand.split(key)
    W1 = jrand.normal(subkey, (128, 256))
    key, subkey = jrand.split(key)
    W2 = jrand.normal(subkey, (256, 64))
    return Params(W1, W2)

def shard_params(params: Params) -> Params:
    W1, W2 = params
    W1 = einshard(W1, 'x y -> x y*')
    W2 = einshard(W2, 'y z -> y* z')
    return Params(W1, W2)

@jax.jit
def forward(params: Params, X: Array) -> Array:
    W1, W2 = params
    Y = X @ W1
    Y = nn.relu(Y)
    Z = Y @ W2
    return Z

def main() -> None:
    key = jrand.key(42)

    # initialize params
    key, subkey = jrand.split(key)
    params = init_params(key)

    # generate random inputs
    key, subkey = jrand.split(key)
    X = jrand.normal(subkey, (8, 128))

    # shard the model params
    params = shard_params(params)
    # replicate the inputs
    X = einshard(X, '... -> * ...')

    out = forward(params, X)
    print(out.shape)  # (8, 64)

if __name__ == '__main__':
    main()
```

The expected output of the model can vary depending on the sharding strategy you use. Any approach is feasible as long as it produces your desired result. However, in practice, obtaining a complete result is often the expectation.

## General Approach for Larger Models

When sharding models like MLPs, simply determine the reduced and free axes during linear multiplications as described above. Then, use einshard to specify the desired parallelism method for sharding. This approach can be applied to larger models as well, but each layer's structure needs careful analysis.

For transformers, analyze which axis of the `q`, `k`, `v` arrays in the attention layer should be sharded and apply einshard accordingly. Similarly, shard the appropriate axes in the MLP layer. For other parameters, such as those in embedding or normalization layers, which do not need sharding, use einshard to replicate them. By specifying how to shard or replicate arrays before computation, larger models can also be simply parallelized. As long as the entire computation process is naturally sharded, JAX's features can be leveraged to achieve smooth parallelism.

This method is applicable not only to small models implemented directly in pure JAX but also to larger models such as LLMs, for example, [mistral-v0.2-jax](https://github.com/yixiaoer/mistral-v0.2-jax) is a large model implemented in pure JAX that uses einshard. It is also applicable to models loaded from existing libraries like Hugging Face, for instance, in the [tpu-training-example](https://github.com/yixiaoer/tpu-training-example), which demonstrates using a Hugging Face model, each part of the model is sharded or replicated using einshard.

With einshard, you can easily implement data parallelism and 1-D tensor parallelism in JAX.
