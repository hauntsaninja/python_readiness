# python_readiness

Are your dependencies ready for new Python?

## Installation

```bash
pip install python_readiness
```

Alternatives include:
- `uvx python_readiness`
- It's a single file script that contains PEP 723 metadata

## Usage

Check if a specific package is ready for a specific Python:
```bash
python_readiness -p numpy --python 3.11
```

This will print the requirement you need to ensure that numpy supports that Python:
```bash
λ python_readiness -p numpy --python 3.13
numpy>=2.1.0      # has_classifier_and_explicit_wheel
λ python_readiness -p numpy --python 3.12
numpy>=1.26.0     # has_classifier_and_explicit_wheel
λ python_readiness -p numpy --python 3.11
numpy>=1.23.3     # has_classifier_and_explicit_wheel
λ python_readiness -p 'numpy>=2' --python 3.11
numpy>=2          # has_classifier_and_explicit_wheel (existing requirement ensures support)
```

Check if a requirements file is ready for a specific Python:
```bash
python_readiness -r requirements.txt --python 3.13
```

This will output new requirements that ensure your environment is restricted to versions that
will support the specified Python version. In particular, look at lines containing "previously".
These are the minimum versions you will need to upgrade to for support.

I find this really useful for updating constraints files when incrementally upgrading a large
codebase to a new Python.

Check if your current environment is ready for the latest Python:
```bash
python_readiness
```

Check if another virtual environment is ready for the latest Python:
```bash
python_readiness -e path/to/.venv
```

See all options:
```bash
python_readiness --help
```

## What are the exact definitions of readiness this uses?

Take a look at the code, in particular `support_from_files`.

It's primarily based on wheel tags and classifiers, but looks at some other metadata and has a
few interesting tricks.

`python_readiness` currently classifies package versions as one of the following levels of support:

Explicitly supported:
- `has_classifier_and_explicit_wheel`

  Both `has_classifier` and `has_explicit_wheel` are true.
- `has_classifier`

  Has a trove classifier for the corresponding Python version.
- `has_explicit_wheel`

  Has a wheel that specifically supports the corresponding Python version (includes abi3 wheels targeting specifically that Python)
- `is_requires_python_lower_bound`

  Niche, but if `Requires-Python: >=3.9` then the package explicitly supports 3.9.

Potentially supported:
- `has_viable_wheel`.

  Has a wheel that is installable on the Python version, but has no explicit indication that upstream has ever run it with that Python version.
- `totally_unknown`

  This is usually because the package only provides an sdist.

Unsupported:
- `unsupported`

Also, if you're interested in looking at the code — the bisection code we use to find earliest
supported versions is interesting, since it can handle non-monotonic support, e.g. in the case of
backports.
