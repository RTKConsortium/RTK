#include "itkHisImageIOFactory.h"

#include <fstream>

//====================================================================
itk::HisImageIOFactory::HisImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "HisImageIO",
                         "His Image IO",
                         1,
                         itk::CreateObjectFunction<HisImageIO>::New() );
}
