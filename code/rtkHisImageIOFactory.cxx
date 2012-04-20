#include "rtkHisImageIOFactory.h"

#include <fstream>

//====================================================================
rtk::HisImageIOFactory::HisImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "HisImageIO",
                         "His Image IO",
                         1,
                         itk::CreateObjectFunction<HisImageIO>::New() );
}
