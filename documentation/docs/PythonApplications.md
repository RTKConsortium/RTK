# Using RTK Applications in Python

RTK applications can be accessed directly in Python. This allows you to use the same functionality as the command-line applications but with the flexibility of Python scripting.

## General Usage

Each application accepts either:

1. **A single string argument**: This mimics the command-line usage.
2. **Keyword arguments**: This provides a more Pythonic interface.

```python
rtk.<application_name>(<arguments>)
```
Only file paths are supported as input—ITK objects cannot be passed directly.

## Example: Running `rtkfdk`

### Command-Line:
```bash
rtkfdk -p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --dimension 256,256,256
```

### Python Equivalent:
```python
rtk.rtkfdk(
    "-p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --dimension 256"
)
```

You can also use keyword arguments for better readability:

```python
rtk.rtkfdk(
    path=".",
    regexp="projections.mha",
    output="fdk.mha",
    geometry="geometry.xml",
    spacing="2,2,2",
    dimension=[256,256,256] # You can use a string or a list
)
```

## Notes

- The Python interface dynamically maps the RTK applications to Python functions.
- **help(rtk.<application_name>)** can be used to see all required and optional arguments.
- Positional arguments are not handled.