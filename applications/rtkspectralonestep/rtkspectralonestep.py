#!/usr/bin/env python
import argparse
import itk
import numpy as np
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="One-step spectral reconstruction (Python port)"
    )
    parser.add_argument(
        "--geometry", "-g", required=True, help="XML geometry file name"
    )
    parser.add_argument("--output", "-o", required=True, help="Output file name")
    parser.add_argument(
        "--niterations", "-n", type=int, default=1, help="Number of iterations"
    )
    parser.add_argument("--input", "-i", help="Material volumes initial guess")
    parser.add_argument(
        "--spectral",
        "-s",
        required=True,
        help="Spectral projections, i.e. photon counts",
    )
    parser.add_argument(
        "--detector", "-d", required=True, help="Detector response file"
    )
    parser.add_argument(
        "--incident", required=True, help="Incident spectrum file (mhd image)"
    )
    parser.add_argument(
        "--attenuations", "-a", required=True, help="Material attenuations file"
    )
    parser.add_argument(
        "--mask",
        "-m",
        help="Apply a support binary mask: reconstruction kept null outside",
        default=None,
    )
    parser.add_argument(
        "--regul_spatial_weights",
        help="Spatial regularization weights file",
        default=None,
    )
    parser.add_argument(
        "--projection_weights", help="Projection weights file", default=None
    )
    parser.add_argument(
        "--thresholds",
        "-t",
        type=float,
        nargs="+",
        required=True,
        help="Lower threshold of bins, expressed in pulse height",
    )
    parser.add_argument(
        "--subsets",
        type=int,
        default=1,
        help="Number of subsets of projections (should not exceed 6)",
    )
    parser.add_argument(
        "--regul_weights",
        type=float,
        nargs="+",
        help="Regularization parameters for each material",
    )
    parser.add_argument(
        "--regul_radius",
        type=int,
        nargs="+",
        help="Radius of the neighborhood for regularization",
    )
    parser.add_argument(
        "--reset_nesterov",
        type=int,
        default=1,
        help="Reset Nesterov after a number of subsets",
    )

    rtk.add_rtkiterations_group(parser)
    rtk.add_rtkprojectors_group(parser)

    return parser


def GetFileHeader(filename: str):
    io = itk.ImageIOFactory.CreateImageIO(filename, itk.CommonEnums.IOFileMode_ReadMode)
    if not io:
        raise RuntimeError(
            f"ImageIOFactory could not create an ImageIO for '{filename}'"
        )
    io.SetFileName(filename)
    io.ReadImageInformation()
    return io


def spectral_bin_detector_response(drm_img, thresholds):
    # drm_img: itk image 2D (energies, pulseHeights)
    region = drm_img.GetLargestPossibleRegion()
    size = region.GetSize()
    numberOfEnergies = size[0]

    numberOfSpectralBins = len(thresholds) - 1

    binnedResponse = np.zeros((numberOfSpectralBins, numberOfEnergies), dtype=float)

    indexDet = itk.Index[2]()
    for energy in range(numberOfEnergies):
        indexDet[0] = energy
        for bin in range(numberOfSpectralBins):
            # First and last couple of values:
            # use trapezoidal rule with linear interpolation
            infPulse = int(np.floor(thresholds[bin]))
            if infPulse < 1:
                raise RuntimeError(f"Threshold {thresholds[bin]} below 0 keV")

            supPulse = int(np.floor(thresholds[bin + 1]))
            if float(supPulse) == thresholds[bin + 1]:
                supPulse -= 1

            if supPulse - infPulse < 3:
                raise RuntimeError("Thresholds are too close for the current code.")

            wInf = infPulse + 1.0 - thresholds[bin]
            indexDet[1] = infPulse - 1  # Index 0 is 1 keV
            binnedResponse[bin, energy] += (
                0.5 * wInf * wInf * drm_img.GetPixel(indexDet)
            )

            indexDet[1] += 1
            binnedResponse[bin, energy] += (
                0.5 * (1.0 + wInf * (2.0 - wInf)) * drm_img.GetPixel(indexDet)
            )

            wSup = thresholds[bin + 1] - supPulse
            indexDet[1] = supPulse  # Index 0 is 1 keV
            if supPulse >= drm_img.GetLargestPossibleRegion().GetSize()[1]:
                raise RuntimeError(
                    f"Threshold {thresholds[bin+1]} above max {drm_img.GetLargestPossibleRegion().GetSize()[1]}"
                )
            binnedResponse[bin, energy] += (
                0.5 * wSup * wSup * drm_img.GetPixel(indexDet)
            )

            indexDet[1] -= 1
            binnedResponse[bin, energy] += (
                0.5 * (1.0 + wSup * (2.0 - wSup)) * drm_img.GetPixel(indexDet)
            )

            # Intermediate values
            for pulseHeight in range(infPulse + 1, supPulse - 1):
                indexDet[1] = pulseHeight
                binnedResponse[bin, energy] += drm_img.GetPixel(indexDet)
    rows, cols = binnedResponse.shape
    v = itk.vnl_matrix[itk.F](rows, cols)
    for i in range(rows):
        row = binnedResponse[i]
        for j in range(cols):
            v.put(i, j, row[j])
    return v


def process(args_info: argparse.Namespace):
    dataType = itk.F
    Dimension = 3

    headerInputMeasuredProjections = GetFileHeader(args_info.spectral)
    headerAttenuations = GetFileHeader(args_info.attenuations)
    nBins = headerInputMeasuredProjections.GetNumberOfComponents()
    nMaterials = headerAttenuations.GetDimensions(0)

    # Define types for the input images
    MeasuredProjectionsType = itk.Image[itk.Vector[dataType, nBins], Dimension]
    MaterialVolumesType = itk.Image[itk.Vector[dataType, nMaterials], Dimension]
    IncidentSpectrumType = itk.Image[dataType, Dimension]
    DetectorResponseType = itk.Image[dataType, Dimension - 1]
    MaterialAttenuationsType = itk.Image[dataType, Dimension - 1]

    # Instantiate and update the readers
    mea = itk.ImageFileReader[MeasuredProjectionsType].New()
    mea.SetFileName(args_info.spectral)
    mea.Update()
    mea = mea.GetOutput()

    incidentSpectrum = itk.ImageFileReader[IncidentSpectrumType].New()
    incidentSpectrum.SetFileName(args_info.incident)
    incidentSpectrum.Update()
    incidentSpectrum = incidentSpectrum.GetOutput()

    detectorResponse = itk.ImageFileReader[DetectorResponseType].New()
    detectorResponse.SetFileName(args_info.detector)
    detectorResponse.Update()
    detectorResponse = detectorResponse.GetOutput()

    materialAttenuations = itk.ImageFileReader[MaterialAttenuationsType].New()
    materialAttenuations.SetFileName(args_info.attenuations)
    materialAttenuations.Update()
    materialAttenuations = materialAttenuations.GetOutput()

    # Read Support Mask if given
    if args_info.mask:
        supportmask = itk.imread(args_info.mask)

    # Read spatial regularization weights if given
    if args_info.regul_spatial_weights:
        spatialRegulWeighs = itk.imread(args_info.regul_spatial_weights)

    # Read projections weights if given
    if args_info.projection_weights:
        projectionWeights = itk.imread(args_info.projection_weights)

    # Create input: either an existing volume read from a file or a blank image
    if args_info.input is not None:
        input = itk.ImageFileReader[MaterialVolumesType].New()
        input.SetFileName(args_info.input)
        input.Update()
        input = input.GetOutput()
    else:
        constantImageSource = itk.ConstantImageSource[MaterialVolumesType].New()
        rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)
        input = constantImageSource.GetOutput()

    # Read the material attenuations image as a matrix (C++ style)
    indexMat = itk.Index[2]()
    nEnergies = materialAttenuations.GetLargestPossibleRegion().GetSize()[1]
    materialAttenuationsMatrix = itk.vnl_matrix[itk.F](nEnergies, nMaterials)
    for energy in range(nEnergies):
        indexMat[1] = energy
        for material in range(nMaterials):
            indexMat[0] = material
            materialAttenuationsMatrix.put(
                energy, material, materialAttenuations.GetPixel(indexMat)
            )

    thresholds = list(args_info.thresholds)
    MaximumPulseHeight = detectorResponse.GetLargestPossibleRegion().GetSize()[1]
    thresholds.append(MaximumPulseHeight)
    if len(thresholds) - 1 != nBins:
        raise RuntimeError(
            f"Number of thresholds {len(thresholds) - 1} does not match the number of bins {nBins}"
        )

    # Read the detector response image as a matrix, and bin it
    drm = spectral_bin_detector_response(detectorResponse, thresholds)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry from {args_info.geometry} ...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Read the regularization parameters
    regulRadius = itk.Size[3]()
    if args_info.regul_radius:
        for i in range(3):
            regulRadius[i] = args_info.regul_radius[
                min(i, len(args_info.regul_radius) - 1)
            ]
    else:
        regulRadius.Fill(0)

    regulWeights = itk.Vector[dataType, nMaterials]()
    if args_info.regul_weights:
        for i in range(nMaterials):
            regulWeights[i] = args_info.regul_weights[
                min(i, len(args_info.regul_weights) - 1)
            ]
    else:
        regulWeights.Fill(0.0)

    if hasattr(itk, "CudaImage"):
        CudaMeasuredProjectionsType = itk.CudaImage[
            itk.Vector[dataType, nBins], Dimension
        ]
        CudaMaterialVolumesType = itk.CudaImage[
            itk.Vector[dataType, nMaterials], Dimension
        ]
        CudaIncidentSpectrumType = itk.CudaImage[dataType, Dimension]

        mechlemOneStep = rtk.MechlemOneStepSpectralReconstructionFilter[
            CudaMaterialVolumesType,
            CudaMeasuredProjectionsType,
            CudaIncidentSpectrumType,
        ].New()

        mechlemOneStep.SetInputMaterialVolumes(itk.cuda_image_from_image(input))
        mechlemOneStep.SetInputIncidentSpectrum(
            itk.cuda_image_from_image(incidentSpectrum)
        )
        mechlemOneStep.SetInputMeasuredProjections(itk.cuda_image_from_image(mea))
        if args_info.mask:
            mechlemOneStep.SetSupportMask(itk.cuda_image_from_image(supportmask))
        if args_info.regul_spatial_weights:
            mechlemOneStep.SetSpatialRegularizationWeights(
                itk.cuda_image_from_image(spatialRegulWeighs)
            )
        if args_info.projection_weights:
            mechlemOneStep.SetProjectionWeights(
                itk.cuda_image_from_image(projectionWeights)
            )
    else:
        mechlemOneStep = rtk.MechlemOneStepSpectralReconstructionFilter[
            MaterialVolumesType, MeasuredProjectionsType, IncidentSpectrumType
        ].New()

        mechlemOneStep.SetInputMaterialVolumes(input)
        mechlemOneStep.SetInputIncidentSpectrum(incidentSpectrum)
        mechlemOneStep.SetInputMeasuredProjections(mea)
        if args_info.mask:
            mechlemOneStep.SetSupportMask(supportmask)
        if args_info.regul_spatial_weights:
            mechlemOneStep.SetSpatialRegularizationWeights(spatialRegulWeighs)
        if args_info.projection_weights:
            mechlemOneStep.SetProjectionWeights(projectionWeights)

    rtk.SetIterationsReportFromArgParse(args_info, mechlemOneStep)
    rtk.SetForwardProjectionFromArgParse(args_info, mechlemOneStep)
    rtk.SetBackProjectionFromArgParse(args_info, mechlemOneStep)

    mechlemOneStep.SetBinnedDetectorResponse(drm)
    mechlemOneStep.SetMaterialAttenuations(materialAttenuationsMatrix)
    mechlemOneStep.SetNumberOfIterations(args_info.niterations)
    mechlemOneStep.SetNumberOfSubsets(args_info.subsets)
    mechlemOneStep.SetRegularizationRadius(regulRadius)
    mechlemOneStep.SetRegularizationWeights(regulWeights)
    if args_info.reset_nesterov:
        mechlemOneStep.SetResetNesterovEvery(args_info.reset_nesterov)
    mechlemOneStep.SetGeometry(geometry)

    mechlemOneStep.Update()

    # Write output
    WriterType = itk.ImageFileWriter[MaterialVolumesType]
    writer = WriterType.New()
    writer.SetFileName(args_info.output)
    writer.SetInput(mechlemOneStep.GetOutput())
    writer.SetImageIO(itk.MetaImageIO.New())
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
