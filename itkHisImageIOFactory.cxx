#ifndef ITKHISIMAGEIOFACTORY_CXX
#define ITKHISIMAGEIOFACTORY_CXX

#include "itkHisImageIOFactory.h"

#include <fstream>

//====================================================================
itk::HisImageIOFactory::HisImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "HisImageIO",
                         "His Image IO",
                         1,
                         itk::CreateObjectFunction<HisImageIO>::New());
}


#endif /* end #define ITKHISIMAGEIOFACTORY_CXX */

