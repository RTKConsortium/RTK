import re
import argparse
from itk import RTK as rtk
import difflib
import inspect

__all__ = [
    "RTKArgumentParser"
]

class RTKHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = rtk.version() + "\n\nusage: "
        return super()._format_usage(usage, actions, groups, prefix)

class RTKArgumentParser(argparse.ArgumentParser):
    def __init__(self, description=None, **kwargs):
        super().__init__(description=description, **kwargs)
        self.formatter_class = RTKHelpFormatter
        # allow negative numeric tokens to be treated as values, not options. This mirrors CPython behavior in python 3.14
        self._negative_number_matcher = re.compile(r'-\.?\d')
        self.add_argument('-V', '--version', action='version', version=rtk.version())

    def build_signature(self) -> inspect.Signature:
        """Build a compact Python signature: only required kwargs + **kwargs."""
        required_params = []
        for action in self._actions:
            name = getattr(action, 'dest', None)
            if not name or name in ("help", "version"):
                continue
            if getattr(action, 'required', False):
                required_params.append(
                    inspect.Parameter(name=name, kind=inspect.Parameter.KEYWORD_ONLY)
                )
        # Add catch-all for the many optional CLI options to keep help inline
        var_kw = inspect.Parameter('kwargs', kind=inspect.Parameter.VAR_KEYWORD)
        return inspect.Signature(required_params + [var_kw])

    def build_usage_examples(self, app_name: str | None = None) -> str:
        """Return a Usage examples block for Python help()."""
        name = app_name or self.prog
        # Collect required destinations
        req = [a.dest for a in self._actions
               if getattr(a, 'required', False) and a.dest and a.dest not in ("help", "version")]
        shell = f"{name}(\"{' '.join([f'--{d} {d.upper()}' for d in req])}\")"
        py = f"{name}({', '.join([f'{d}={d.upper()}' for d in req])})"
        return (
            "Usage:\n"
            f"    • Shell-style: {shell}\n"
            f"    • Python API:  {py}\n\n"
        )

    def apply_signature(self, func):
        """Apply the built signature to a callable and return it."""
        func.__signature__ = self.build_signature()
        return func

    def parse_args(self, args=None, namespace=None):
        """Parse args with optional single-token comma list support for multi-value options.
        Supported forms:
          --opt A B C      (space separated)
          --opt A,B,C      (single token, comma separated)
        """
        neutralized = {}
        for action in self._actions:
            dest = getattr(action, 'dest', None)
            if not dest or dest in ("help", "version"):
                continue
            nargs = getattr(action, 'nargs', None)
            if nargs != '+':
                continue
            t = getattr(action, 'type', None)
            if t and t is not str:
                neutralized[dest] = t
                action.type = str
        namespace = super().parse_args(args, namespace)
        for action in self._actions:
            dest = getattr(action, 'dest', None)
            if dest not in neutralized:
                continue
            caster = neutralized[dest]
            val = getattr(namespace, dest, None)
            if isinstance(val, list):
                # Case 1: user supplied a single token containing commas (e.g. "1,2,3").
                # Split on commas, strip whitespace, drop empty pieces, then cast each element.
                if len(val) == 1 and isinstance(val[0], str) and ',' in val[0]:
                    pieces = [s for s in (p.strip() for p in val[0].split(',')) if s]
                    setattr(namespace, dest, [caster(p) for p in pieces])
                else:
                    # Case 2: normal space-separated form (e.g. "1 2 3"). Just cast every token.
                    setattr(namespace, dest, [caster(tk) for tk in val])
            action.type = neutralized[dest]
        return namespace

    def parse_kwargs(self, func_name: str | None = None, **kwargs):
        """Convert Python kwargs to argv and parse them.
        Lists/tuples for multi-value options are serialized as a single comma token."""
        actions = {a.dest: a for a in self._actions if a.dest and a.dest not in ("help", "version")}
        for key in kwargs:
            if key not in actions:
                matches = difflib.get_close_matches(key, actions.keys(), n=3, cutoff=0.5)
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
            opt_strings = list(action.option_strings)
            flag = next((o for o in opt_strings if o.startswith('--')), opt_strings[0])
            if isinstance(val, bool):
                if val:
                    argv.append(flag)
            elif isinstance(val, (list, tuple)):
                argv += [flag, ",".join(map(str, val))]
            else:
                argv += [flag, str(val)]
        return self.parse_args(argv)
