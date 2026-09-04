import pytest
import itk
from itk import RTK as rtk


@pytest.fixture()
def parser():
    p = rtk.RTKArgumentParser()
    p.add_argument("--string-single", type=str, required=True)
    p.add_argument("--number-single", type=int, required=True)
    p.add_argument("--string-many", type=str, nargs="+", required=True)
    p.add_argument("--number-many", type=int, nargs="+", required=True)
    return p


def assert_args(args):
    assert args.string_single == "a"
    assert args.number_single == 1
    assert args.string_many == ["a", "b", "c"]
    assert args.number_many == [1, 2, 3]


def test_cli_comma_separated(parser):
    args = parser.parse_args(
        [
            "--string-single",
            "a",
            "--number-single",
            "1",
            "--string-many",
            "a,b,c",
            "--number-many",
            "1,2,3",
        ]
    )
    assert_args(args)


def test_cli_space_separated(parser):
    args = parser.parse_args(
        [
            "--string-single",
            "a",
            "--number-single",
            "1",
            "--string-many",
            "a",
            "b",
            "c",
            "--number-many",
            "1",
            "2",
            "3",
        ]
    )
    assert_args(args)


def test_python_api(parser):
    args = parser.parse_kwargs(
        string_single="a",
        number_single=1,
        string_many=["a", "b", "c"],
        number_many=[1, 2, 3],
    )
    assert_args(args)


def test_python_api_comma_string(parser):
    args = parser.parse_kwargs(
        string_single="a", number_single=1, string_many="a,b,c", number_many="1,2,3"
    )
    assert_args(args)
