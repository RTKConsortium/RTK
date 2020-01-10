/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

//#ifndef RTKGGOFUNCTIONS_H
//#define RTKGGOFUNCTIONS_H

#ifndef rtkGgoFunctions_h
#define rtkGgoFunctions_h

#include "rtkMacro.h"
#include "rtkConstantImageSource.h"
#include "rtkIOFactories.h"
#include "rtkProjectionsReader.h"
#include <itkRegularExpressionSeriesFileNames.h>
#include <itksys/RegularExpression.hxx>

namespace rtk
{

/** \brief Create 3D image from gengetopt specifications.
 *
 * This function sets a ConstantImageSource object from command line options stored in ggo struct.
 *  The image is not buffered to allow streaming. The image is filled with zeros.
 *  The required options in the ggo struct are:
 *     - dimension: image size in pixels
 *     - spacing: image spacing in coordinate units
 *     - origin: image origin in coordinate units
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
template <class TConstantImageSourceType, class TArgsInfo>
void
SetConstantImageSourceFromGgo(typename TConstantImageSourceType::Pointer source, const TArgsInfo & args_info)
{
  using ImageType = typename TConstantImageSourceType::OutputImageType;

  const unsigned int Dimension = ImageType::GetImageDimension();

  typename ImageType::SizeType imageDimension;
  imageDimension.Fill(args_info.dimension_arg[0]);
  for (unsigned int i = 0; i < std::min(args_info.dimension_given, Dimension); i++)
    imageDimension[i] = args_info.dimension_arg[i];

  typename ImageType::SpacingType imageSpacing;
  imageSpacing.Fill(args_info.spacing_arg[0]);
  for (unsigned int i = 0; i < std::min(args_info.spacing_given, Dimension); i++)
    imageSpacing[i] = args_info.spacing_arg[i];

  typename ImageType::PointType imageOrigin;
  for (unsigned int i = 0; i < Dimension; i++)
    imageOrigin[i] = imageSpacing[i] * (imageDimension[i] - 1) * -0.5;
  for (unsigned int i = 0; i < std::min(args_info.origin_given, Dimension); i++)
    imageOrigin[i] = args_info.origin_arg[i];

  typename ImageType::DirectionType imageDirection;
  if (args_info.direction_given)
    for (unsigned int i = 0; i < Dimension; i++)
      for (unsigned int j = 0; j < Dimension; j++)
        imageDirection[i][j] = args_info.direction_arg[i * Dimension + j];
  else
    imageDirection.SetIdentity();

  source->SetOrigin(imageOrigin);
  source->SetSpacing(imageSpacing);
  source->SetDirection(imageDirection);
  source->SetSize(imageDimension);
  source->SetConstant(0.);

  // Copy output image information from an existing file, if requested
  // Overwrites parameters given in command line, if any
  if (args_info.like_given)
  {
    using LikeReaderType = itk::ImageFileReader<ImageType>;
    typename LikeReaderType::Pointer likeReader = LikeReaderType::New();
    likeReader->SetFileName(args_info.like_arg);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(likeReader->UpdateOutputInformation());
    source->SetInformationFromImage(likeReader->GetOutput());
  }

  TRY_AND_EXIT_ON_ITK_EXCEPTION(source->UpdateOutputInformation());
}

/** \brief Read a stack of 2D projections from gengetopt specifications.
 *
 * This function sets a ProjectionsReader object from command line options stored in ggo struct.
 * The projections are not buffered to allow streaming.
 * The required options in the ggo struct are:
 *     - verbose
 *     - path: path containing projections
 *     - regexp: regular expression to select projection files in path
 *     - nsort: boolean to (des-)activate the numeric sort for expression matches
 *     - submatch: index of the submatch that will be used to sort matches
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
template <class TArgsInfo>
const std::vector<std::string>
GetProjectionsFileNamesFromGgo(const TArgsInfo & args_info)
{
  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(args_info.nsort_flag);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(args_info.submatch_arg);

  if (args_info.verbose_flag)
    std::cout << "Regular expression matches " << names->GetFileNames().size() << " file(s)..." << std::endl;

  // Check submatch in file names
  if (args_info.submatch_given)
  {
    // Check that the submatch number returns something relevant
    itksys::RegularExpression reg;
    if (!reg.compile(args_info.regexp_arg))
    {
      itkGenericExceptionMacro(<< "Error compiling regular expression " << args_info.regexp_arg);
    }

    // Store the full filename and the selected sub expression match
    for (const std::string & name : names->GetFileNames())
    {
      reg.find(name);
      if (reg.match(args_info.submatch_arg) == std::string(""))
      {
        itkGenericExceptionMacro(<< "Cannot find submatch " << args_info.submatch_arg << " in " << name
                                 << " from regular expression " << args_info.regexp_arg);
      }
    }
  }

  std::vector<std::string> fileNames = names->GetFileNames();
  rtk::RegisterIOFactories();
  std::vector<size_t> idxtopop;
  size_t              i = 0;
  for (const auto & fn : fileNames)
  {
    itk::ImageIOBase::Pointer imageio =
#if (ITK_VERSION_MAJOR == 5) && (ITK_VERSION_MINOR >= 1)
      itk::ImageIOFactory::CreateImageIO(fn.c_str(), itk::ImageIOFactory::FileModeEnum::ReadMode);
#else
      itk::ImageIOFactory::CreateImageIO(fn.c_str(), itk::ImageIOFactory::FileModeType::ReadMode);
#endif

    if (imageio.IsNull())
    {
      std::cerr << "Ignoring file: " << fn << "\n";
      idxtopop.push_back(i);
    }
    i++;
  }
  std::reverse(idxtopop.begin(), idxtopop.end());
  for (const auto & id : idxtopop)
  {
    fileNames.erase(fileNames.begin() + id);
  }

  return fileNames;
}

template <class TProjectionsReaderType, class TArgsInfo>
void
SetProjectionsReaderFromGgo(typename TProjectionsReaderType::Pointer reader, const TArgsInfo & args_info)
{
  const std::vector<std::string> fileNames = GetProjectionsFileNamesFromGgo(args_info);

  // Vector component extraction
  if (args_info.component_given)
  {
    reader->SetVectorComponent(args_info.component_arg);
  }

  // Change image information
  const unsigned int Dimension = TProjectionsReaderType::OutputImageType::GetImageDimension();
  typename TProjectionsReaderType::OutputImageDirectionType direction;
  if (args_info.newdirection_given)
  {
    direction.Fill(args_info.newdirection_arg[0]);
    for (unsigned int i = 0; i < args_info.newdirection_given; i++)
      direction[i / Dimension][i % Dimension] = args_info.newdirection_arg[i];
    reader->SetDirection(direction);
  }
  typename TProjectionsReaderType::OutputImageSpacingType spacing;
  if (args_info.newspacing_given)
  {
    spacing.Fill(args_info.newspacing_arg[0]);
    for (unsigned int i = 0; i < args_info.newspacing_given; i++)
      spacing[i] = args_info.newspacing_arg[i];
    reader->SetSpacing(spacing);
  }
  typename TProjectionsReaderType::OutputImagePointType origin;
  if (args_info.neworigin_given)
  {
    direction.Fill(args_info.neworigin_arg[0]);
    for (unsigned int i = 0; i < args_info.neworigin_given; i++)
      origin[i] = args_info.neworigin_arg[i];
    reader->SetOrigin(origin);
  }

  // Crop boundaries
  typename TProjectionsReaderType::OutputImageSizeType upperCrop, lowerCrop;
  upperCrop.Fill(0);
  lowerCrop.Fill(0);
  for (unsigned int i = 0; i < args_info.lowercrop_given; i++)
    lowerCrop[i] = args_info.lowercrop_arg[i];
  reader->SetLowerBoundaryCropSize(lowerCrop);
  for (unsigned int i = 0; i < args_info.uppercrop_given; i++)
    upperCrop[i] = args_info.uppercrop_arg[i];
  reader->SetUpperBoundaryCropSize(upperCrop);

  // Conditional median
  typename TProjectionsReaderType::MedianRadiusType medianRadius;
  medianRadius.Fill(0);
  for (unsigned int i = 0; i < args_info.radius_given; i++)
    medianRadius[i] = args_info.radius_arg[i];
  reader->SetMedianRadius(medianRadius);
  if (args_info.multiplier_given)
    reader->SetConditionalMedianThresholdMultiplier(args_info.multiplier_arg);

  // Shrink / Binning
  typename TProjectionsReaderType::ShrinkFactorsType binFactors;
  binFactors.Fill(1);
  for (unsigned int i = 0; i < args_info.binning_given; i++)
    binFactors[i] = args_info.binning_arg[i];
  reader->SetShrinkFactors(binFactors);

  // Boellaard scatter correction
  if (args_info.spr_given)
    reader->SetScatterToPrimaryRatio(args_info.spr_arg);
  if (args_info.nonneg_given)
    reader->SetNonNegativityConstraintThreshold(args_info.nonneg_arg);
  if (args_info.airthres_given)
    reader->SetAirThreshold(args_info.airthres_arg);

  // I0 and IDark
  if (args_info.i0_given)
    reader->SetI0(args_info.i0_arg);
  reader->SetIDark(args_info.idark_arg);

  // Line integral flag
  if (args_info.nolineint_flag)
    reader->ComputeLineIntegralOff();

  // Water precorrection
  if (args_info.wpc_given)
  {
    std::vector<double> coeffs;
    coeffs.assign(args_info.wpc_arg, args_info.wpc_arg + args_info.wpc_given);
    reader->SetWaterPrecorrectionCoefficients(coeffs);
  }

  // Pass list to projections reader
  reader->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(reader->UpdateOutputInformation());
}

/** \brief Set the correct RTK backprojection type from gengetopt
 * Given a gengetopt bp_arg option from the rtkprojectors_section.ggo, will
 * set the corresponding enum value from rtk::IterativeConeBeamReconstructionFilter
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
template <class TArgsInfo, class TIterativeReconstructionFilter>
void
SetBackProjectionFromGgo(const TArgsInfo & args_info, TIterativeReconstructionFilter * recon)
{
  switch (args_info.bp_arg)
  {
    case (-1): // bp__NULL
      recon->SetBackProjectionFilter(TIterativeReconstructionFilter::BP_UNKNOWN);
      break;
    case (0): // bp_arg_VoxelBasedBackProjection
      recon->SetBackProjectionFilter(TIterativeReconstructionFilter::BP_VOXELBASED);
      break;
    case (1): // bp_arg_Joseph
      recon->SetBackProjectionFilter(TIterativeReconstructionFilter::BP_JOSEPH);
      break;
    case (2): // bp_arg_CudaVoxelBased
      recon->SetBackProjectionFilter(TIterativeReconstructionFilter::BP_CUDAVOXELBASED);
      break;
    case (3): // bp_arg_CudaRayCast
      recon->SetBackProjectionFilter(TIterativeReconstructionFilter::BP_CUDARAYCAST);
      break;
    case (4): // bp_arg_JosephAttenuated
      recon->SetBackProjectionFilter(TIterativeReconstructionFilter::BP_JOSEPHATTENUATED);
      break;
    case (5): // bp_arg_RotationBased
      recon->SetBackProjectionFilter(TIterativeReconstructionFilter::BP_ZENG);
      break;
  }
}

/** \brief Set the correct RTK forward projection type from gengetopt
 * Given a gengetopt fp_arg option from the rtkprojectors_section.ggo, will
 * set the corresponding enum value from rtk::IterativeConeBeamReconstructionFilter
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
template <class TArgsInfo, class TIterativeReconstructionFilter>
void
SetForwardProjectionFromGgo(const TArgsInfo & args_info, TIterativeReconstructionFilter * recon)
{
  switch (args_info.fp_arg)
  {
    case (-1): // fp__NULL
      recon->SetForwardProjectionFilter(TIterativeReconstructionFilter::FP_UNKNOWN);
      break;
    case (0): // fp_arg_Joseph
      recon->SetForwardProjectionFilter(TIterativeReconstructionFilter::FP_JOSEPH);
      break;
    case (1): // fp_arg_CudaRayCast
      recon->SetForwardProjectionFilter(TIterativeReconstructionFilter::FP_CUDARAYCAST);
      break;
    case (2): // fp_arg_JosephAttenuated
      recon->SetForwardProjectionFilter(TIterativeReconstructionFilter::FP_JOSEPHATTENUATED);
      break;
    case (3): // fp_arg_RotationBased
      recon->SetForwardProjectionFilter(TIterativeReconstructionFilter::FP_ZENG);
      break;
  }
}

} // namespace rtk

#endif // rtkGgoFunctions_h
