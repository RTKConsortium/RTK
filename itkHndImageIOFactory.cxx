#include "itkHndImageIOFactory.h"

#include <fstream>

//====================================================================
itk::HndImageIOFactory::HndImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "HndImageIO",
                         "Hnd Image IO",
                         1,
                         itk::CreateObjectFunction<HndImageIO>::New() );
}
