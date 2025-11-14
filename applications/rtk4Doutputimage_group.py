def add_rtk4Doutputimage_group(parser):
    group = parser.add_argument_group("Output 4D image properties")
    group.add_argument(
        "--origin", help="Origin (default=centered)", type=float, nargs="+"
    )
    group.add_argument(
        "--dimension",
        help="Dimension (Deprecated) Use --size instead.",
        type=int,
        nargs="+",
        default=[256],
    )
    group.add_argument("--size", help="Size", type=int, nargs="+", default=[256])
    group.add_argument("--spacing", help="Spacing", type=float, nargs="+", default=[1])
    group.add_argument("--direction", help="Direction", type=float, nargs="+")
    group.add_argument(
        "--frames",
        help="Number of time frames in the 4D reconstruction",
        type=int,
        default=10,
    )
    group.add_argument(
        "--like",
        help="Copy information from this image (origin, size, spacing, direction, frames)",
        type=str,
    )
