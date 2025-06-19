import argparse
from itk import RTK as rtk

__all__ = [
    "rtk_argument_parser"
]

def rtk_argument_parser(desc):
    return argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=rtk.version()
    )
