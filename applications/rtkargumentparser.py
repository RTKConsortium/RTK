import argparse
from gettext import gettext as _
from itk import RTK as rtk

__all__ = [
    "rtk_argument_parser"
]

class RTKHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = _(rtk.version() + "\n\nusage: ")
        return super()._format_usage(usage, actions, groups, prefix)

def rtk_argument_parser(desc):

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=RTKHelpFormatter
    )
    parser.add_argument('-V', '--version', action='version', version=rtk.version())

    return parser
