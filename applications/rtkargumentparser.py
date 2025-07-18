import argparse
from itk import RTK as rtk

__all__ = [
    "rtk_argument_parser"
]

def rtk_argument_parser(desc):
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=rtk.version()
    )
    parser.add_argument('-V', '--version', action='version', version=rtk.version())
    return parser
