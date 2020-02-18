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

#include "rtkDCMImagXImageIOFactory.h"

//====================================================================
rtk::DCMImagXImageIOFactory::DCMImagXImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "DCMImagXImageIO",
                         "ImagX Image IO for its DICOM file format",
                         true,
                         itk::CreateObjectFunction<DCMImagXImageIO>::New());
}

// Undocumented API used to register during static initialization.
// DO NOT CALL DIRECTLY.

namespace itk
{

static bool DCMImagXImageIOFactoryHasBeenRegistered;

void RTK_EXPORT
     DCMImagXImageIOFactoryRegister__Private()
{
  if (!DCMImagXImageIOFactoryHasBeenRegistered)
  {
    DCMImagXImageIOFactoryHasBeenRegistered = true;
    rtk::DCMImagXImageIOFactory::RegisterOneFactory();
  }
}

} // end namespace itk
