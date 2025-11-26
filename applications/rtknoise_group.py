import itk

__all__ = [
    "add_rtknoise_group",
    "SetNoiseFromArgParse",
]


# Mimics rtknoise_section.ggo
def add_rtknoise_group(parser):
    rtknoise_group = parser.add_argument_group("Noise")
    rtknoise_group.add_argument(
        "--gaussian",
        help="Gaussian noise parameters: <mean> Noise level and <std> Noise standard deviation",
        type=float,
        nargs="+",
    )
    rtknoise_group.add_argument(
        "--poisson",
        help=(
            "Poisson noise parameters: <I0> Number of impinging photons per pixel "
            "and <muref> reference linear attenuation coefficient"
        ),
        type=float,
        nargs="+",
    )


def SetNoiseFromArgParse(projections, args_info):
    OutputImageType = type(projections)
    output = projections

    if args_info.gaussian is not None:
        noisy = itk.AdditiveGaussianNoiseImageFilter[
            OutputImageType, OutputImageType
        ].New()
        mean, std = args_info.gaussian
        noisy.SetInput(output)
        noisy.SetMean(mean)
        noisy.SetStandardDeviation(std)
        noisy.UpdateOutputInformation()
        output = noisy.GetOutput()

    if args_info.poisson is not None:
        I0, muref = args_info.poisson

        multiply = itk.MultiplyImageFilter[
            OutputImageType, OutputImageType, OutputImageType
        ].New()
        multiply.SetInput(output)
        multiply.SetConstant(-muref)

        expf = itk.ExpImageFilter[OutputImageType, OutputImageType].New()
        expf.SetInput(multiply.GetOutput())

        multiply2 = itk.MultiplyImageFilter[
            OutputImageType, OutputImageType, OutputImageType
        ].New()
        multiply2.SetInput(expf.GetOutput())
        multiply2.SetConstant(I0)

        poisson = itk.ShotNoiseImageFilter[OutputImageType].New()
        poisson.SetInput(multiply2.GetOutput())

        threshold = itk.ThresholdImageFilter[OutputImageType].New()
        threshold.SetInput(poisson.GetOutput())
        threshold.SetLower(1.0)
        threshold.SetOutsideValue(1.0)

        multiply3 = itk.MultiplyImageFilter[
            OutputImageType, OutputImageType, OutputImageType
        ].New()
        multiply3.SetInput(threshold.GetOutput())
        multiply3.SetConstant(1.0 / I0)

        logf = itk.LogImageFilter[OutputImageType, OutputImageType].New()
        logf.SetInput(multiply3.GetOutput())

        multiply4 = itk.MultiplyImageFilter[
            OutputImageType, OutputImageType, OutputImageType
        ].New()
        multiply4.SetInput(logf.GetOutput())
        multiply4.SetConstant(-1.0 / muref)

        multiply4.UpdateOutputInformation()
        output = multiply4.GetOutput()

    return output
