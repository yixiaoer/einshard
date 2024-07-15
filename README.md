# Einshard

High-level array sharding API for JAX

## Introduction

`einshard` is a Python library designed to simplify the process of sharding and replicating arrays in JAX. `einshard` enables integration of various parallelism techniques without modifying the model code. Whether working with simple models like MLPs or larger models like LLMs, `einshard` provides a solution for distributing computations across multiple devices. This library allows users to define sharding strategies with simple expressions.

This project originated as a part of the [Mistral 7B v0.2 JAX](https://github.com/yixiaoer/mistral-v0.2-jax) project and has since evolved into an independent project.

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Installation

This library requires at least Python 3.12.

You need to have JAX installed by [choosing the correct installation method](https://jax.readthedocs.io/en/latest/installation.html) before installing einshard.

After JAX is installed, install einshard with this command:

```sh
pip install einshard
```

## Documentation

Please see the detailed documentation at <https://einshard.readthedocs.io/en/latest/>.

We have a [draft PDF file](https://www.overleaf.com/read/qgrbbtrntwrg) for the theory.
