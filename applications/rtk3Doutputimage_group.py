import itk

__all__ = [
    'add_rtk3Doutputimage_group',
    'SetConstantImageSourceFromArgParse',
]

# Mimicks rtk3Doutputimage_section.ggo
def add_rtk3Doutputimage_group(parser):
  rtk3Doutputimage_group = parser.add_argument_group('Output 3D image properties')
  rtk3Doutputimage_group.add_argument('--origin', help='Origin (default=centered)', type=float, nargs='+')
  rtk3Doutputimage_group.add_argument('--dimension', help='Dimension', type=int, nargs='+', default=[256])
  rtk3Doutputimage_group.add_argument('--spacing', help='Spacing', type=float, nargs='+', default=[1])
  rtk3Doutputimage_group.add_argument('--direction', help='Direction', type=float, nargs='+')
  rtk3Doutputimage_group.add_argument('--like', help='Copy information from this image (origin, dimension, spacing, direction)')

# Mimicks SetConstantImageSourceFromGgo
def SetConstantImageSourceFromArgParse(source, args_info):
  ImageType = type(source.GetOutput())

  Dimension = ImageType.GetImageDimension()

  imageDimension = itk.Size[Dimension]()
  imageDimension.Fill(args_info.dimension[0])
  for i in range(min(len(args_info.dimension), Dimension)):
    imageDimension[i] = args_info.dimension[i]

  imageSpacing = itk.Vector[itk.D, Dimension]()
  imageSpacing.Fill(args_info.spacing[0])
  for i in range(min(len(args_info.spacing), Dimension)):
    imageSpacing[i] = args_info.spacing[i]

  imageOrigin = itk.Point[itk.D, Dimension]()
  for i in range(Dimension):
    imageOrigin[i] = imageSpacing[i] * (imageDimension[i] - 1) * -0.5
  if args_info.origin is not None:
    for i in range(min(len(args_info.origin), Dimension)):
      imageOrigin[i] = args_info.origin[i]

  imageDirection = source.GetOutput().GetDirection()
  if args_info.direction is not None:
    for i in range(Dimension):
      for j in range(Dimension):
        imageDirection[i][j] = args_info.direction[i * Dimension + j]

  source.SetOrigin(imageOrigin)
  source.SetSpacing(imageSpacing)
  source.SetDirection(imageDirection)
  source.SetSize(imageDimension)
  source.SetConstant(0.)

  # Copy output image information from an existing file, if requested
  # Overwrites parameters given in command line, if any
  if args_info.like is not None:
    LikeReaderType = itk.ImageFileReader[ImageType]
    likeReader = LikeReaderType.New()
    likeReader.SetFileName(args_info.like)
    likeReader.UpdateOutputInformation()
    source.SetInformationFromImage(likeReader.GetOutput())

  source.UpdateOutputInformation()
