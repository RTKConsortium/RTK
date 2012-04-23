#include "rtkHndImageIOFactory.h"

#include <fstream>

//====================================================================
rtk::HndImageIOFactory::HndImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "HndImageIO",
                         "Hnd Image IO",
                         1,
                         itk::CreateObjectFunction<HndImageIO>::New() );
}
