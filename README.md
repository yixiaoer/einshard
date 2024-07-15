# Einshard

High-level array sharding API for JAX

## Introduction

`einshard` is a Python library designed to simplify the process of sharding and replicating arrays in JAX. `einshard` enables integration of various parallelism techniques without modifying the model code. Whether working with simple models like MLPs or larger models like LLMs, `einshard` provides a solution for distributing computations across multiple devices. This library allows users to define sharding strategies with simple expressions.

This project originated as a part of the [Mistral 7B v0.2 JAX](https://github.com/yixiaoer/mistral-v0.2-jax) project and has since evolved into an independent project.

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Installation

This library requires at least Python 3.12.

```sh
pip install einshard
```

You need to have JAX installed by [choosing the correct installation method](https://jax.readthedocs.io/en/latest/installation.html) before installing Einshard.

## Syntax and Usage

`einshard` leverages a simple and clear method to shard and replicate a single array according to the specified einshard expression. How to write a correct einshard expression? Let's take a look at the syntax.

An einshard expression consists of three parts, the left-hand side, the arrow, and the right-hand side:

* The left-hand side includes all the axes names of the array. 
    * Different axis names should be separated by spaces (either multiple or single spaces are allowed, but a single space is recommended).
    * Axis names are composed of case-sensitive letters.

* The arrow, represented by `->`, separates the left and right sides. Spaces around the arrow are optional, but a single space is recommended.

* The right-hand side includes all the axes names of the array along with the corresponding sharding and replication methods.

Here's how to specify the sharding and replication methods. The examples assume the usage of 16 devices.

**Test Initialization:**

To ensure JAX uses the CPU backend with 16 devices, initialize it before the actual code:

```python
n_devices = 16
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + f' --xla_force_host_platform_device_count={n_devices}'

from einshard import einshard 
import jax
import jax.numpy as jnp
```

1. **Using numbers to specify sharding or replication**:

   When specifying axis sharding, add the number directly after the axis name (no spaces). If no number is specified, the default is 1, meaning the axis is not sharded.
   
   For replication, write the standalone number. If no standalone number is specified, the array is not replicated.

   ```python
   X = jnp.zeros((4, 8))
   X = einshard(X, 'a b -> 2 a2 b4')
   ```

   This indicates that the `a` axis is split into 2 parts, the `b` axis into 4 parts, and the entire array is replicated 2 times.

2. **Using `...` to omit unsharded axes**:

   If some axes do not need sharding, use `...` in both the left and right sides to omit them.

   ```python
   X = jnp.zeros((4, 8, 20, 10))
   X = einshard(X, 'a ... d -> 4 a2 ... d2')
   X = einshard(X, 'a b c d -> 4 a2 b c d2')  # the same
   ```

3. **Using `*` to determine the number of devices dynamically**:

   Instead of hardcoding numbers, use star `*` to determine the number of devices dynamically. The `*` specifies a ratio. The actual number of devices will be determined proportionally based on the total number of devices.

   ```python
   X = jnp.zeros((4, 8))
   X = einshard(X, 'a b -> * a2 b')
   jax.debug.visualize_array_sharding(X)
   ```
   ```text
    ┌────────────────────────────────────────────────┐
    │                                                │
    │             CPU 0,2,4,6,8,10,12,14             │
    │                                                │
    │                                                │
    ├────────────────────────────────────────────────┤
    │                                                │
    │             CPU 1,3,5,7,9,11,13,15             │
    │                                                │
    │                                                │
    └────────────────────────────────────────────────┘
   ```

   In the above example, the `a` axis is split into 2 parts. Since there are 16 devices in total, the remaining `*` part will determine the remaining device number, resulting in 8 replicates of the array.

   You can also introduce multiple `*` in an einshard expression, allowing for flexible and proportional number of the remaining devices.

   ```python
   X = jnp.zeros((4, 8))
   X = einshard(X, 'a b -> 2 a2* b*')
   jax.debug.visualize_array_sharding(X)
   ```

   ```text
    ┌───────────────────────┬───────────────────────┐
    │        CPU 0,8        │        CPU 1,9        │
    ├───────────────────────┼───────────────────────┤
    │       CPU 2,10        │       CPU 3,11        │
    ├───────────────────────┼───────────────────────┤
    │       CPU 4,12        │       CPU 5,13        │
    ├───────────────────────┼───────────────────────┤
    │       CPU 6,14        │       CPU 7,15        │
    └───────────────────────┴───────────────────────┘
   ```

   This expression indicates the array is replicated 2 times, and the remaining 8 devices are proportionally split based on the numbers before the `*`. Specifically, the `a` axis is split into 4 parts, and the `b` axis is split into 2 parts, following the 2:1 ratio specified by the numbers before the `*`.

4. **Placement of numbers and stars**:

   How to determine where to place numbers and stars? As previously explained, add numbers or `*` directly after the axis name for sharding, and write standalone numbers or `*` for replication. Does the order in the einshard expression matter?

   First, any standalone numbers represent the array's replication count. If there are multiple standalone numbers, their product represents the total replication count.

   The position of these numbers or `*` can affect how devices are allocated to different partitions.

   TODO: More examples and edge cases will be added to illustrate different placements and their effects on device allocation.

When using `einshard`, consider the number of devices you have. You should always use `*` and `...` when possible to simplify the expressions and leverage the library's flexibility.

By following these syntax rules, `einshard` allows you to efficiently implement data parallelism and 1-D tensor parallelism in JAX, among other parallel techniques, without modifying your model code.

## Application

With `einshard`, you can easily shard arrays across different devices in JAX without modifying your model code, simplifying parallel computation in JAX.

Using the `einshard` API, you can partition or replicate a single array according to the specified einshard expression and distribute it across various devices. If the computation is naturally-sharded, JAX, leveraging the XLA compiler, will automatically determine the optimal output array sharding based on the input array's sharding. This means that by setting up the input array's sharding as desired, JAX will handle the parallel computation accordingly and automatically. This allows you to shard a model across different devices by placing the model’s arrays using `einshard` before computation begins.

### Steps for Sharding a Model with `einshard`

1. **Analyze the Model Structure**:

   Determine which axes are reduced axes and which are free axes in the arrays used for matrix multiplication (as can be represented by einsum expressions).

2. **Determine Sharding and Replication Strategy**:

   For each array in the computation, decide how to shard or replicate each axis to ensure the entire computation process remains naturally sharded and the final output matches your expectations.

3. **Select Parallelism Method and Apply `einshard`**:
   
   Use `einshard` to specify the sharding or replication of the arrays based on your parallelism strategy.


### Example: Sharding a 2-Layer MLP

Consider a 2-layer MLP. The computations can be represented by the following einsum expressions (ignoring the activation function, which is point-wise and not considered for sharding):

* First matrix multiplication: `bx, xy -> by`, to compute `X @ W1 = Y`.
* Second matrix multiplication: `by, yz -> bz`, to compute `Y @ W2 = Z`.

Step-by-Step Sharding:

1. **Analyze the model structure**:
   - Input array \( X \) has shape \( bx \) and multiplies with \( W1 \) (shape \( xy \)) to produce \( Y \) (shape \( by \)). Here, \( x \) is a reduced axis, while \( b \) and \( y \) are free axes.
   - \( Y \) (shape \( by \)) then multiplies with \( W2 \) (shape \( yz \)) to produce \( Z \) (shape \( bz \)). In this computation, \( y \) is a reduced axis, while \( b \) and \( z \) are free axes.

2. **Determine Sharding and Replication**:

   The array \( X \) is the input data, while \( W1 \) and \( W2 \) are the model parameters for this 2-layer MLP. These arrays need to be sharded before the computation starts. Consider the following potential sharding strategies:

   * Sharding the `b` axis in the first matrix computation results in the `b` axis being sharded in \( Y \), and consequently, the final output \( Z \) will also have the `b` axis sharded. The same applies if the `z` axis is sharded.

   * Sharding the `x` axis in the first matrix computation results in \( Y \) being complete after all-reduce. If the `z` axis of \( W2 \) is sharded, the output \( Z \) will be sharded.

   * Sharding the `y` axis in the first matrix computation results in the `y` axis being sharded in \( Y \). If the `y` axis of \( W2 \) is also sharded, the output \( Z \) will be complete after all-reduce.

3. **Use `einshard` for sharding or replication**:
   If you expect to apply 1-D tensor parallelism and ensure the model parameters are efficiently sharded while obtaining a complete result, you can choose to shard the \( y \) axis of \( W1 \) and \( W2 \):

   ```python
   w1_proj = einshard(w1_proj, 'x y -> x y*')
   w2_proj = einshard(w2_proj, 'y z -> y* z')
   ```

The expected output of the model can vary depending on the sharding strategy you use. Any approach is feasible as long as it produces your desired result. However, in practice, obtaining a complete result is often the expectation.


### General Approach for Larger Models

When sharding models like MLPs, simply determine the reduced and free axes during linear multiplications as described above. Then, use `einshard` to specify the desired parallelism method for sharding. This approach can be applied to larger models as well, but each layer's structure needs careful analysis.


For transformers, analyze which axis of the `q`, `k`, `v` arrays in the attention layer should be sharded and apply `einshard` accordingly. Similarly, shard the appropriate axes in the MLP layer. For other parameters, such as those in embedding or normalization layers, which do not need sharding, use `einshard` to replicate them. By specifying how to shard or replicate arrays before computation, larger models can also be effectively parallelized. As long as the entire computation process is naturally sharded, JAX's features can be leveraged to achieve smooth parallelism.

This method is applicable not only to small models implemented directly in pure JAX but also to larger models such as LLMs, for example, [mistral-v0.2-jax](https://github.com/yixiaoer/mistral-v0.2-jax) is a large model implemented in pure JAX that uses `einshard`. It is also applicable to models loaded from existing libraries like Hugging Face, for instance, in the [tpu-training-example](https://github.com/yixiaoer/tpu-training-example), which demonstrates using a Hugging Face model, each part of the model is sharded or replicated using `einshard`.

With `einshard`, you can efficiently implement multiple parallelisms in JAX for effective parallel computing.

## Development

Crente venv:

```sh
python3.12 -m venv venv
. venv/bin/activate
```

Install dependencies:

```sh
pip install -U pip
pip install -U wheel
pip install "jax[cpu]"
pip install -r requirements.txt
```

Run test:

```sh
python tests/test_einshard.py
```

Build package:

```sh
pip install build
python -m build
```

Build docs:

```sh
cd docs
make html
```

```sh
cd docs/_build/html
python -m http.server -b 127.0.0.1
```
