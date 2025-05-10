import itk
from itk import RTK as rtk
import inspect
import difflib
from typing import Any, Callable
import argparse


# Write a 3D circular projection geometry to a file.
def write_geometry(geometry, filename):
    # Create and configure the writer
    writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    writer.SetObject(geometry)
    writer.SetFilename(filename)

    # Write the geometry to file
    writer.WriteFile()


# Read a 3D circular projection geometry from a file.
def read_geometry(filename):
    # Create and configure the reader
    reader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    reader.SetFilename(filename)
    reader.GenerateOutputInformation()

    # Return the geometry object
    return reader.GetOutputObject()


# Returns the progress percentage
class PercentageProgressCommand:
    def __init__(self, caller):
        self.percentage = -1
        self.caller = caller

    def callback(self):
        new_percentage = int(self.caller.GetProgress() * 100)
        if new_percentage > self.percentage:
            print(
                f"\r{self.caller.GetNameOfClass()} {new_percentage}% completed.",
                end="",
                flush=True,
            )
            self.percentage = new_percentage

    def End(self):
        print()  # Print newline when execution ends


# Returns a lambda function that parses a comma-separated string and converts each element to the specified type.
def comma_separated_args(value_type):
    def parser(value):
        try:
            return [value_type(s.strip()) for s in value.split(",")]
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"invalid {value_type.__name__} value: {value!r}"
            )

    return parser


def parse_kwargs(parser: argparse.ArgumentParser, func_name: str = None, **kwargs: Any):
    """
    Convert kwargs → argv → parsed Namespace.
    Raises TypeError for unknown keys (with suggestions),
    ValueError for parsing errors.
    """
    # collect valid destinations
    valid = {
        action.dest
        for action in parser._actions
        if action.dest and action.dest not in ("help", "version")
    }

    # early reject unknown keys
    for key in kwargs:
        if key not in valid:
            matches = difflib.get_close_matches(key, valid, n=3, cutoff=0.5)
            name = func_name or parser.prog or "function"
            msg = f"{name}() got an unexpected keyword argument '{key}'"
            if matches:
                msg += f"\nDid you mean: {', '.join(matches)} ?"
            else:
                msg += f"\nValid arguments are: {', '.join(sorted(valid))}"
            raise TypeError(msg)

    # convert kwargs to argv
    argv = []
    for key, val in kwargs.items():
        flag = f"--{key}" if len(key) > 1 else f"-{key}"
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, (list, tuple)):
            for v in val:
                argv += [flag, str(v)]
        else:
            argv += [flag, str(val)]

    return parser.parse_args(argv)


def patch_signature(func: Callable[..., Any], parser: argparse.ArgumentParser):
    """
    Set func.__signature__ based on parser._actions, so inspect.signature(func)
    shows the real keyword-only parameters (with defaults).
    Also replaces func.__doc__ with the parser's help text.
    """
    params = []
    for action in parser._actions:
        name = action.dest
        if not name or name in ("help", "version"):
            continue
        if getattr(action, "required", False):
            default = inspect._empty
        else:
            if action.default is not None:
                default = action.default
            else:
                default = inspect._empty

        params.append(
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
            )
        )

    func.__signature__ = inspect.Signature(params)
