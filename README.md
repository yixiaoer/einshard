# Einshard

High-level array sharding API for JAX

## Installation

This library requires at least Python 3.12.

```sh
pip install einshard
```

## Usage

```python
# initialising JAX CPU backend with 16 devices
n_devices = 16
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + f' --xla_force_host_platform_device_count={n_devices}'

# actual code starts here
from einshard import einshard
import jax
import jax.numpy as jnp

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

## Development

```sh
python3.12 -m venv venv
. venv/bin/activate
```

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
