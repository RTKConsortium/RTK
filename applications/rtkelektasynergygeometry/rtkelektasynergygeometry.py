#!/usr/bin/env python
import argparse
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Creates an RTK geometry file from an Elekta Synergy acquisition."
    )

    parser.add_argument("--image-db", "-i", help="Image table filename (prior to XVI5)")
    parser.add_argument("--frame-db", "-f", help="Frame table filename (prior to XVI5)")
    parser.add_argument("--xml", "-x", help="XML file name (starting with XVI5)")
    parser.add_argument("--dicom-uid", "-u", help="Dicom uid of the acquisition")
    parser.add_argument("--output", "-o", help="Output file name", required=True)

    return parser


def process(args_info: argparse.Namespace):

    if (
        args_info.image_db is not None
        and args_info.frame_db is not None
        and args_info.dicom_uid is not None
        and args_info.xml is None
    ):
        if args_info.verbose:
            print("Building geometry from DBF tables...")
        reader = rtk.ElektaSynergyGeometryReader.New()
        reader.SetDicomUID(args_info.dicom_uid)
        reader.SetImageDbfFileName(args_info.image_db)
        reader.SetFrameDbfFileName(args_info.frame_db)
        reader.UpdateOutputData()
        geometry = reader.GetGeometry()
    elif (
        args_info.image_db
        and args_info.frame_db
        and args_info.dicom_uid
        and args_info.xml is not None
    ):
        if args_info.verbose:
            print(f"Reading geometry information from {args_info.xml}...")
        geometryReader = rtk.ElektaXVI5GeometryXMLFileReader.New()
        geometryReader.SetFilename(args_info.xml)
        geometryReader.GenerateOutputInformation()
        geometry = geometryReader.GetGeometry()
    else:
        raise ValueError(
            "You must either provide image_db, frame_db and dicom_uid for versions up to v4 or xml starting with v5."
        )

    if args_info.verbose:
        print(f"Writing geometry to {args_info.output}...")
    rtk.write_geometry(geometry, args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
