# Einshard

A high-level array sharding API for JAX

## Installation

This library requires at least Python 3.12.

```sh
pip install einshard
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

Testing:

```sh
python tests/test_einshard.py
```
