#ifndef RTKGGOFUNCTIONS_H
#define RTKGGOFUNCTIONS_H

#include "rtkMacro.h"
#include "rtkConstantImageSource.h"

namespace rtk
{

/** This function set a ConstantImageSource object from command line options stored in ggo struct.
    The image is not buffered to allow streaming. The image is filled with zeros.
    The required options in the ggo struct are:
      - dimension: image size in pixels
      - spacing: image spacing in coordinate units
      - origin: image origin in coordinate units
*/
template< class TConstantImageSourceType, class TArgsInfo >
void
SetConstantImageSourceFromGgo(typename TConstantImageSourceType::Pointer source, const TArgsInfo &args_info)
{
  typedef typename TConstantImageSourceType::OutputImageType ImageType;
  
  const unsigned int Dimension = ImageType::GetImageDimension();

  typename ImageType::SizeType imageDimension;
  imageDimension.Fill(args_info.dimension_arg[0]);
  for(unsigned int i=0; i<vnl_math_min(args_info.dimension_given, Dimension); i++)
    imageDimension[i] = args_info.dimension_arg[i];

  typename ImageType::SpacingType imageSpacing;
  imageSpacing.Fill(args_info.spacing_arg[0]);
  for(unsigned int i=0; i<vnl_math_min(args_info.spacing_given, Dimension); i++)
    imageSpacing[i] = args_info.spacing_arg[i];

  typename ImageType::PointType imageOrigin;
  for(unsigned int i=0; i<Dimension; i++)
    imageOrigin[i] = imageSpacing[i] * (imageDimension[i]-1) * -0.5;
  for(unsigned int i=0; i<vnl_math_min(args_info.origin_given, Dimension); i++)
    imageOrigin[i] = args_info.origin_arg[i];

  source->SetOrigin( imageOrigin );
  source->SetSpacing( imageSpacing );
  source->SetSize( imageDimension );
  source->SetConstant( 0. );
  source->UpdateOutputInformation();
}

}

#endif // RTKGGOFUNCTIONS_H
