import itk

__all__ = [
    "add_rtknoise_group",
    "AddNoiseFromArgParse",
]


# Mimics rtknoise_section.ggo
def add_rtknoise_group(parser):
    rtknoise_group = parser.add_argument_group("Poisson and Gaussian noise")
    rtknoise_group.add_argument(
        "--poisson",
        help="Comma-separated pre-log Poisson noise parameters: I0 pre-log normalization and (optional) pre-exp multiplicative factor (default 0.01879 mm^-1, attenuation coefficient of water at 75 keV)",
        type=float,
        nargs="+",
    )
    rtknoise_group.add_argument(
        "--gaussian",
        help="Additive Gaussian noise standard deviation (pre-log if --poisson given, without exp/log otherwise)",
        type=float,
    )


def AddNoiseFromArgParse(projections, args_info):
    OutputImageType = type(projections)
    output = projections

    if args_info.poisson is not None:
        I0 = args_info.poisson[0]
        muref = 0.01879  # mm^-1, attenuation coefficient of water at 75 keV
        if len(args_info.poisson) == 2:
            muref = args_info.poisson[1]

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
        noisy = itk.AdditiveGaussianNoiseImageFilter[
            OutputImageType, OutputImageType
        ].New()
        if args_info.gaussian is not None:
            gaussian.SetInput(threshold.GetOutput())
            gaussian.SetStandardDeviation(args_info.gaussian)
            multiply3.SetInput(gaussian.GetOutput())
        else:
            multiply3.SetInput(threshold.GetOutput())
        multiply3.SetConstant(1.0 / I0)

        logf = itk.LogImageFilter[OutputImageType, OutputImageType].New()
        logf.SetInput(multiply3.GetOutput())

        multiply4 = itk.MultiplyImageFilter[
            OutputImageType, OutputImageType, OutputImageType
        ].New()
        multiply4.SetInput(logf.GetOutput())
        multiply4.SetConstant(-1.0 / muref)

        multiply4.Update()
        output = multiply4.GetOutput()
    elif args_info.gaussian is not None:
        gaussian = itk.AdditiveGaussianNoiseImageFilter[
            OutputImageType, OutputImageType
        ].New()
        gaussian.SetInput(output)
        gaussian.SetStandardDeviation(args_info.gaussian)
        gaussian.Update()
        output = gaussian.GetOutput()

    return output
