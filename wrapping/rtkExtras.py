import itk
from itk import RTK as rtk
import inspect
import difflib
from typing import Any, Callable
import argparse
import importlib
import shlex


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
    ValueError for parsing errors, and TypeError for incorrect types.
    """
    # Collect valid destinations and their expected types
    valid = {
        action.dest: action.type
        for action in parser._actions
        if action.dest and action.dest not in ("help", "version")
    }

    # Early reject unknown keys
    for key in kwargs:
        if key not in valid:
            matches = difflib.get_close_matches(key, valid.keys(), n=3, cutoff=0.5)
            name = func_name or parser.prog or "function"
            msg = f"{name}() got an unexpected keyword argument '{key}'"
            if matches:
                msg += f"\nDid you mean: {', '.join(matches)}?"
            else:
                msg += f"\nValid arguments are: {', '.join(sorted(valid.keys()))}"
            raise TypeError(msg)

    # Convert kwargs to argv
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
    shows the real keyword-only parameters.
    Required arguments are listed first.
    """
    required_params = []
    optional_params = []

    for action in parser._actions:
        name = action.dest
        if not name or name in ("help", "version"):
            continue

        if getattr(action, "required", False):
            default = inspect._empty
            required_params.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                )
            )
        else:
            if action.default is not None:
                default = action.default
            else:
                default = inspect._empty

            optional_params.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                )
            )

    # Combine required and optional parameters, with required ones first
    func.__signature__ = inspect.Signature(required_params + optional_params)


def make_application_func(app_name):
    def application_func(*args, **kwargs):
        # Dynamically import the application's module
        app_module = importlib.import_module(f"itk.{app_name}")

        main = getattr(app_module, "main", None)
        process = getattr(app_module, "process", None)
        build_parser = getattr(app_module, "build_parser", None)

        parser = build_parser()

        # Shell style
        if len(args) == 1 and isinstance(args[0], str) and not kwargs:
            argv = shlex.split(args[0])
            try:
                return main(argv)
            except Exception as e:
                print(e)
                return

        # Handle positional arguments
        if len(args) > 0:
            raise ValueError(f"The positional arguments are not supported. ")

        # Python API style
        if len(args) == 0:
            parser = build_parser()
            parsed = rtk.parse_kwargs(parser, func_name=app_name, **kwargs)
            return process(parsed)

    # Patch the function's signature using the parser
    parser = importlib.import_module(f"itk.{app_name}").build_parser()
    patch_signature(application_func, parser)

    # Update the function's metadata to reflect the application name
    application_func.__name__ = app_name
    application_func.__module__ = "itk.RTK"

    # Extract required arguments
    required_args = []
    for action in parser._actions:
        if action.required:
            arg_name = f"--{action.dest} {action.dest.upper()}"
            required_args.append(arg_name)

    # Generate shell-style and Python API-style examples
    shell_example = f'{app_name}("{" ".join(required_args)}")'
    python_api_example = f'{app_name}({", ".join([f"{arg.dest}={arg.dest.upper()}" for arg in parser._actions if arg.required])})'

    # Get the help text and remove the 'usage:' line(s)
    help_lines = parser.format_help().splitlines()
    option_idx = 0
    for i, line in enumerate(help_lines):
        if line.strip().startswith("options:"):
            option_idx = i

    usage_examples = f"""
-----\n
Usage:
    • Shell-style: {shell_example}
    • Python API:  {python_api_example}
"""
    application_func.__doc__ = (
        parser.description + usage_examples + "\n".join(help_lines[option_idx:])
    )

    return application_func
