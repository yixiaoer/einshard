# Development Guide

Crente venv:

```sh
python3.12 -m venv venv
. venv/bin/activate
```

Install the latest version of Mypy. This is because of the [python/mypy#15238](https://github.com/python/mypy/issues/15238) issue.

```sh
pip install -U pip
pip install -U wheel
pip install -U git+https://github.com/python/mypy.git
```

Install dependencies:

```sh
pip install "jax[cpu]"
pip install -r requirements.txt
pip install -r docs/requirements.txt
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
