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

class RTKArgumentParser(rtk.RTKArgumentParser):
    def __init__(self, description=None, **kwargs):
        super().__init__(description=description, **kwargs)
        self.formatter_class = RTKHelpFormatter
        self.add_argument('-V', '--version', action='version', version=rtk.version())