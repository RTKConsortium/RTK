import itk
from itk import RTK as rtk
import importlib
import shlex
import math


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


# Read a signal file
def read_signal_file(filename):
    signal_vector = []
    try:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line:
                    value = float(line)
                    rounded = round(value * 100) / 100
                    if rounded == 1.0:
                        signal_vector.append(0.0)
                    else:
                        signal_vector.append(rounded)
    except OSError:
        raise RuntimeError(f"Could not open signal file {filename}")
    return signal_vector


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


def make_application_func(app_name):
    """
    Factory: returns a function that runs the RTK application `app_name`
    with either Python-style kwargs or a single CLI-style string.
    """

    def app_func(*args, **kwargs):
        app_module = importlib.import_module(f"itk.{app_name}")
        parser = app_module.build_parser()
        # Ensure help/usage shows the logical app name in Python contexts
        parser.prog = app_name

        if kwargs:
            if hasattr(parser, "parse_kwargs"):
                args_ns = parser.parse_kwargs(func_name=app_name, **kwargs)
            else:
                raise TypeError(f"Parser for {app_name} has no parse_kwargs method.")
        elif args and len(args) == 1 and isinstance(args[0], str):
            # Treat single string argument as CLI
            argv = shlex.split(args[0])
            args_ns = parser.parse_args(argv)
        else:
            args_ns = parser.parse_args()

        return app_module.process(args_ns)

    # Metadata for help()
    _parser = importlib.import_module(f"itk.{app_name}").build_parser()
    _parser.prog = app_name
    _parser.apply_signature(app_func)
    app_func.__name__ = app_name
    app_func.__module__ = "itk.RTK"
    # Python-only help: version, description + examples + options header
    description = (_parser.description or "").rstrip()
    examples = _parser.build_usage_examples(app_name)
    options = _parser.format_help()
    idx = options.lower().find("options:")
    opt_text = options[idx:].strip()
    parts = [rtk.version(), description, examples, opt_text]
    app_func.__doc__ = "\n\n".join(parts)

    return app_func
