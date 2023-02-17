import itk

__all__ = [
    'add_rtkinputprojections_group',
    'GetProjectionsFileNamesFromArgParse',
]

# Mimicks rtkinputprojections_section.ggo
def add_rtkinputprojections_group(parser):
  rtkinputprojections_group = parser.add_argument_group('Input projections and their pre-processing')
  rtkinputprojections_group.add_argument('--path', '-p', help='Path containing projections', required=True)
  rtkinputprojections_group.add_argument('--regexp', '-r', help='Regular expression to select projection files in path', required=True)
  rtkinputprojections_group.add_argument('--nsort', help='Numeric sort for regular expression matches', action='store_true')
  rtkinputprojections_group.add_argument('--submatch', help='Index of the submatch that will be used to sort matches', type=int, default=0)
  rtkinputprojections_group.add_argument('--nolineint', help='Disable raw to line integral conversion, just casts to float', type=bool, default=False)
  rtkinputprojections_group.add_argument('--newdirection', help='New value of input projections (before pre-processing)', type=float, nargs='+')
  rtkinputprojections_group.add_argument('--neworigin', help='New origin of input projections (before pre-processing)', type=float, nargs='+')
  rtkinputprojections_group.add_argument('--newspacing', help='New spacing of input projections (before pre-processing)', type=float, nargs='+')
  rtkinputprojections_group.add_argument('--lowercrop', help='Lower boundary crop size', type=int, nargs='+', default=[0])
  rtkinputprojections_group.add_argument('--uppercrop', help='Upper boundary crop size', type=int, nargs='+', default=[0])
  rtkinputprojections_group.add_argument('--binning', help='Shrink / Binning factos in each direction', type=int, nargs='+', default=[1])
  rtkinputprojections_group.add_argument('--wpc', help='Water precorrection coefficients (default is no correction)', type=float, nargs='+')
  rtkinputprojections_group.add_argument('--spr', help='Boellaard scatter correction: scatter-to-primary ratio', type=float, default=0)
  rtkinputprojections_group.add_argument('--nonneg', help='Boellaard scatter correction: non-negativity threshold', type=float)
  rtkinputprojections_group.add_argument('--airthres', help='Boellaard scatter correction: air threshold', type=float)
  rtkinputprojections_group.add_argument('--i0', help='I0 value (when assumed constant per projection), 0 means auto', type=float)
  rtkinputprojections_group.add_argument('--idark', help='IDark value, i.e., value when beam is off', type=float, default=0)
  rtkinputprojections_group.add_argument('--component', help='Vector component to extract, for multi-material projections', type=int, default=0)
  rtkinputprojections_group.add_argument('--radius', help='Radius of neighborhood for conditional median filtering', type=int, nargs='+', default=[0])
  rtkinputprojections_group.add_argument('--multiplier', help='Threshold multiplier for conditional median filtering', type=float, default=0)

# Mimicks GetProjectionsFileNamesFromGgo
def GetProjectionsFileNamesFromArgParse(args_info):
  # Generate file names
  names = itk.RegularExpressionSeriesFileNames.New()
  names.SetDirectory(args_info.path)
  names.SetNumericSort(args_info.nsort)
  names.SetRegularExpression(args_info.regexp)
  names.SetSubMatch(args_info.submatch)

  if args_info.verbose:
    print(f'Regular expression matches {len(names.GetFileNames())} file(s)...')

  # Check submatch in file names TODO

  fileNames = names.GetFileNames()
  # rtk.RegisterIOFactories() TODO
  idxtopop = []
  i = 0
  for fn in fileNames:
    imageio = itk.ImageIOFactory.CreateImageIO(fn, itk.CommonEnums.IOFileMode_ReadMode)
    if imageio is None:
      print(f'Ignoring file: {fn}')
      idxtopop.append(i)
    ++i

  for id in idxtopop:
    fileNames.pop(id)

  return fileNames

# Mimicks SetProjectionsReaderFromGgo
def SetProjectionsReaderFromArgParse(reader, args_info):
  fileNames = GetProjectionsFileNamesFromArgParse(args_info)

  # Vector component extraction
  if args_info.component is not None:
    reader.SetVectorComponent(args_info.component)

  # Change image information
  Dimension = reader.GetOutput().GetImageDimension()
  if args_info.newdirection is not None:
    direction = itk.Matrix[itk.D,Dimension,Dimension]()
    direction.Fill(args_info.newdirection[0])
    for i in range(len(args_info.newdirection)):
      direction[i / Dimension][i % Dimension] = args_info.newdirection_arg[i]
    reader.SetDirection(direction)

  if args_info.newspacing is not None:
    spacing = itk.Vector[itk.D,Dimension]()
    spacing.Fill(args_info.newspacing[0])
    for i in range(len(args_info.newspacing)):
      spacing[i] = args_info.newspacing[i]
    reader.SetSpacing(spacing)

  if args_info.neworigin is not None:
    origin = itk.Point[itk.D,Dimension]()
    origin.Fill(args_info.neworigin[0])
    for i in range(len(args_info.neworigin)):
      origin[i] = args_info.neworigin[i]
    reader.SetOrigin(origin)

  # Crop boundaries
  upperCrop = [0]*Dimension
  lowerCrop = [0]*Dimension
  if args_info.lowercrop is not None:
    for i in range(len(args_info.lowercrop)):
      lowerCrop[i] = args_info.lowercrop[i]
  reader.SetLowerBoundaryCropSize(lowerCrop)
  if args_info.uppercrop is not None:
    for i in range(len(args_info.uppercrop)):
      upperCrop[i] = args_info.uppercrop[i]
  reader.SetUpperBoundaryCropSize(upperCrop)

  # Conditional median
  medianRadius = reader.GetMedianRadius()
  if args_info.radius is not None:
    for i in range(len(args_info.radius)):
      medianRadius[i] = args_info.radius[i]
  reader.SetMedianRadius(medianRadius)
  if args_info.multiplier is not None:
    reader.SetConditionalMedianThresholdMultiplier(args_info.multiplier)

  # Shrink / Binning
  binFactors = reader.GetShrinkFactors()
  if args_info.binning is not None:
    for i in range(len(args_info.binning)):
       binFactors[i] = args_info.binning[i]
  reader.SetShrinkFactors(binFactors)

  # Boellaard scatter correction
  if args_info.spr is not None:
    reader.SetScatterToPrimaryRatio(args_info.spr)
  if args_info.nonneg is not None:
    reader.SetNonNegativityConstraintThreshold(args_info.nonneg)
  if args_info.airthres is not None:
    reader.SetAirThreshold(args_info.airthres)

  # I0 and IDark
  if args_info.i0 is not None:
    reader.SetI0(args_info.i0)
  reader.SetIDark(args_info.idark)

  # Line integral flag
  if args_info.nolineint:
    reader.ComputeLineIntegralOff()

  # Water precorrection
  if args_info.wpc is not None:
    reader.SetWaterPrecorrectionCoefficients(coeffs)

  # Pass list to projections reader
  reader.SetFileNames(fileNames)
  reader.UpdateOutputInformation()
