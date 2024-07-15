# Quick Start

## Installation

This library requires at least Python 3.12.

You need to have JAX installed by [choosing the correct installation method](https://jax.readthedocs.io/en/latest/installation.html) before installing einshard.

After JAX is installed, install einshard with this command:

```sh
pip install einshard
```

## Shard a JAX Array Using Einshard

For testing purpose, we initialise the JAX CPU backend with 16 devices. This should be run before the actual code (e.g. placed at the top of the script):

```python
n_devices = 16
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + f' --xla_force_host_platform_device_count={n_devices}'
```

Import JAX and einshard:

```python
from einshard import einshard 
import jax
import jax.numpy as jnp
```

Code:

```python
a = jnp.zeros((4, 8))
a = einshard(a, 'a b -> * a* b2*')
jax.debug.visualize_array_sharding(a)
```

Output:

```
┌──────────┬──────────┬──────────┬──────────┐
│          │          │          │          │
│ CPU 0,8  │ CPU 1,9  │ CPU 2,10 │ CPU 3,11 │
│          │          │          │          │
│          │          │          │          │
├──────────┼──────────┼──────────┼──────────┤
│          │          │          │          │
│ CPU 4,12 │ CPU 5,13 │ CPU 6,14 │ CPU 7,15 │
│          │          │          │          │
│          │          │          │          │
└──────────┴──────────┴──────────┴──────────┘
```
