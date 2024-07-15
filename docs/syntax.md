# Syntax

Einshard leverages a simple and clear method to shard and replicate a single array according to the specified einshard expression. How to write a correct einshard expression? Let's take a look at the syntax.

## Einshard Expression

An einshard expression consists of two parts, separated by an arrow `->`:

* The left-hand side includes all the axes names of the array. 

    * Different axis names should be separated by spaces (either multiple or single spaces are allowed, but a single space is recommended).

    * Axis names are composed of case-sensitive letters.

* The right-hand side includes all the axes names of the array along with the corresponding sharding and replication methods.

## Sharding and Replication Methods

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

   TODO: More examples and edge cases will be added to illustrate different placements and their effects on device groups of devices.

When using einshard, consider the number of devices you have. You should always use `*` and `...` when possible to simplify the expressions and leverage the library's flexibility.

By following these syntax rules, einshard allows you to efficiently implement data parallelism and 1-D tensor parallelism in JAX without modifying your model code.
