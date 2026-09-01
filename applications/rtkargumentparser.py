import re
import argparse
from itk import RTK as rtk
import difflib
import inspect
from typing import Optional

__all__ = ["RTKArgumentParser"]

"""RTK application argument parser.

Extends ``argparse.ArgumentParser`` so that the same application can be
invoked from the command line *and* from the Python API via keyword
arguments.

Multi-value arguments (``nargs="+"``) can be passed in three ways:

- **Space-separated (CLI):** ``--opt A B C``
- **Comma-separated (CLI):** ``--opt A,B,C``
- **Python list/tuple (API):** ``app(opt=["A", "B", "C"])``

The Python API works by joining lists into a single comma token
(``"A,B,C"``) before calling ``parse_args``, which splits it back into
``["A", "B", "C"]`` and casts each element to the declared type.
"""


def _make_help_formatter(version):
    class Formatter(argparse.ArgumentDefaultsHelpFormatter):
        def _format_usage(self, usage, actions, groups, prefix=None):
            if prefix is None:
                prefix = (version or "") + "\n\nusage: "
            return super()._format_usage(usage, actions, groups, prefix)

    return Formatter


class RTKArgumentParser(argparse.ArgumentParser):
    """Argument parser for RTK Python applications.

    Wraps ``argparse.ArgumentParser`` with:

    - Common options added to every application (``--version``, ``--verbose``).
    - Support for negative numeric tokens as values (e.g. ``--offset -1 -0.5``).
    - Multi-value comma splitting so ``parse_kwargs`` lists round-trip
      correctly through ``parse_args``.

    Use ``parse_kwargs(**kwargs)`` for the Python API and
    ``parse_args(argv)`` for the command line.
    """

    def __init__(self, description=None, version=None, **kwargs):
        super().__init__(description=description, **kwargs)
        self._version = version or rtk.__version__
        self.formatter_class = _make_help_formatter(self._version)
        # allow negative numeric tokens to be treated as values, not options. This mirrors CPython behavior in python 3.14
        self._negative_number_matcher = re.compile(r"-\.?\d")
        # Common options available to all RTK Python applications
        self.add_argument("-V", "--version", action="version", version=self._version)
        self.add_argument(
            "-v", "--verbose", help="Verbose execution", action="store_true"
        )

    def required_dests(self):
        """Return sorted destination names of all required arguments.

        Used internally by ``apply_signature`` and ``build_usage_examples``
        to avoid duplicating the required-actions filter.
        """
        return sorted(
            a.dest
            for a in self._actions
            if getattr(a, "required", False)
            and a.dest
            and a.dest not in ("help", "version")
        )

    def apply_signature(self, func):
        """Apply a compact signature to a callable for help().

        Only required kwargs appear in the signature; optional arguments
        are captured by **kwargs.
        """
        params = [
            inspect.Parameter(name=d, kind=inspect.Parameter.KEYWORD_ONLY)
            for d in self.required_dests()
        ]
        params.append(inspect.Parameter("kwargs", kind=inspect.Parameter.VAR_KEYWORD))
        func.__signature__ = inspect.Signature(params)
        return func

    def build_usage_examples(self, app_name: Optional[str] = None) -> str:
        """Return a usage examples block for Python help().

        Shows both shell-style and Python API examples using only the
        required arguments.
        """
        name = app_name or self.prog
        req = self.required_dests()
        shell = name + "(" + " ".join(f"--{d} {d.upper()}" for d in req) + ")"
        py = name + "(" + ", ".join(f"{d}={d.upper()}" for d in req) + ")"
        return f"Usage:\n    • Shell-style: {shell}\n    • Python API:  {py}\n\n"

    def parse_args(self, args=None, namespace=None):
        """Parse a token list, with comma-splitting for multi-value options.

        For every ``nargs="+"`` argument, this method:
          1. Temporarily sets the type to ``str`` so argparse does not choke
             on a comma token like ``"1,2,3"``.
          2. Calls the base ``argparse.ArgumentParser.parse_args``.
          3. Restores the original types (in a ``finally`` block).
          4. Splits any single comma token into separate elements and casts
             each element back to the original type.

        Accepts:
          - Token list:  ``["--opt", "A", "B", "C"]``
          - Comma token: ``["--opt", "A,B,C"]``

        Do **not** pass a single string — ``parse_args`` expects a
        sequence of tokens, not a shell command string.
        """
        multi_valued = {}
        for action in self._actions:
            dest = getattr(action, "dest", None)
            if not dest or dest in ("help", "version"):
                continue
            if getattr(action, "nargs", None) != "+":
                continue
            # Neutralize all types (including str) so comma tokens like "a,b" or
            # "1,2,3" can be split after parsing. The original type is restored below.
            cast = action.type or str
            multi_valued[dest] = cast
            action.type = str
        try:
            namespace = super().parse_args(args, namespace)
        finally:
            for action in self._actions:
                dest = getattr(action, "dest", None)
                if dest in multi_valued:
                    action.type = multi_valued[dest]
        for dest, cast in multi_valued.items():
            val = getattr(namespace, dest, None)
            if not isinstance(val, list):
                # Option not supplied on this invocation; leave its default/None intact.
                continue
            # Case 1: user supplied a single token containing commas (e.g. "1,2,3" or
            # "a.nrrd,b.nrrd"). Split on commas, strip whitespace, drop empty pieces.
            if len(val) == 1 and isinstance(val[0], str) and "," in val[0]:
                pieces = [s for s in (p.strip() for p in val[0].split(",")) if s]
                setattr(namespace, dest, [cast(piece) for piece in pieces])
            else:
                # Case 2: normal space-separated form (e.g. "1 2 3"). Just cast every token.
                setattr(namespace, dest, [cast(piece) for piece in val])
        return namespace

    def parse_kwargs(self, func_name: Optional[str] = None, **kwargs):
        """Convert Python keyword arguments into a token list and parse them.

        Lists and tuples are serialized as a single comma-separated token
        (e.g. ``["a", "b"]`` → ``"a,b"``) so they round-trip correctly
        through ``parse_args``.

        Examples::

            app(opt=["A", "B", "C"])   # list → comma token → split back
            app(opt=("A", "B"))       # tuple, same behavior
            app(opt="A,B")            # pre-joined string, also works

        Scalar values (str, int, float) and boolean flags are passed through
        directly.  Unknown keyword arguments raise ``TypeError`` with a
        fuzzy "Did you mean …?" suggestion.
        """
        actions = {
            a.dest: a
            for a in self._actions
            if a.dest and a.dest not in ("help", "version")
        }
        for key in kwargs:
            if key not in actions:
                matches = difflib.get_close_matches(
                    key, actions.keys(), n=3, cutoff=0.5
                )
                name = func_name or self.prog or "function"
                msg = f"{name}() got an unexpected keyword argument '{key}'"
                if matches:
                    msg += f"\nDid you mean: {', '.join(matches)}?"
                else:
                    msg += f"\nValid arguments are: {', '.join(sorted(actions.keys()))}"
                raise TypeError(msg)
        argv = []
        for key, val in kwargs.items():
            action = actions[key]
            flag = next(
                (o for o in action.option_strings if o.startswith("--")),
                action.option_strings[0],
            )
            if isinstance(val, bool):
                if val:
                    argv.append(flag)
            elif isinstance(val, (list, tuple)):
                argv += [flag, ",".join(map(str, val))]
            else:
                argv += [flag, str(val)]
        return self.parse_args(argv)
