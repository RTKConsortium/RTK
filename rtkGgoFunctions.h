#ifndef RTKGGOFUNCTIONS_H
#define RTKGGOFUNCTIONS_H

#include "rtkMacro.h"

#include <itkImage.h>

namespace rtk
{

/** This function creates an image from command line options stored in the parameter.
    The image is filled with zeros.
    The required options are:
      - dimension: image size in pixels
      - spacing: image spacing in coordinate units
      - origin: image origin in coordinate units
*/
template< class TImageType, class TArgsInfo >
typename TImageType::Pointer
CreateImageFromGgo(const TArgsInfo &args_info)
{
  const unsigned int Dimension = TImageType::GetImageDimension();

  typename TImageType::SizeType imageDimension;
  imageDimension.Fill(args_info.dimension_arg[0]);
  for(unsigned int i=0; i<vnl_math_min(args_info.dimension_given, Dimension); i++)
    imageDimension[i] = args_info.dimension_arg[i];

  typename TImageType::SpacingType imageSpacing;
  imageSpacing.Fill(args_info.spacing_arg[0]);
  for(unsigned int i=0; i<vnl_math_min(args_info.spacing_given, Dimension); i++)
    imageSpacing[i] = args_info.spacing_arg[i];

  typename TImageType::PointType imageOrigin;
  for(unsigned int i=0; i<Dimension; i++)
    imageOrigin[i] = imageSpacing[i] * (imageDimension[i]-1) * -0.5;
  for(unsigned int i=0; i<vnl_math_min(args_info.origin_given, Dimension); i++)
    imageOrigin[i] = args_info.origin_arg[i];

  typename TImageType::Pointer image = TImageType::New();
  image->SetRegions( imageDimension );
  image->SetOrigin(imageOrigin);
  image->SetSpacing(imageSpacing);
  image->Allocate();
  image->FillBuffer(0.);

  return image;
}

}

#endif // RTKGGOFUNCTIONS_H
