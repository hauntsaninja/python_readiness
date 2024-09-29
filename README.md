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

Check if your current environment is ready for the latest Python:
```bash
python_readiness
```

Check if a specific package is ready for a specific Python:
```bash
python_readiness -p numpy --python 3.11
```

Check if a requirements file is ready for a specific Python:
```bash
python_readiness -r requirements.txt --python 3.13
```
