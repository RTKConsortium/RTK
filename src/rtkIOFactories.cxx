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

#include "rtkIOFactories.h"
#include <itkImageIOFactory.h>
#include <itkGDCMImageIOFactory.h>

// Varian Obi includes
#include "rtkHndImageIOFactory.h"

// Varian ProBeam includes
#include "rtkXimImageIOFactory.h"

// Elekta Synergy includes
#include "rtkHisImageIOFactory.h"

// ImagX includes
#include "rtkImagXImageIOFactory.h"
#include "rtkDCMImagXImageIOFactory.h"

// European Synchrotron Radiation Facility
#include "rtkEdfImageIOFactory.h"

// Xrad small animal scanner
#include "rtkXRadImageIOFactory.h"

// Ora / medPhoton file format
#include "rtkOraImageIOFactory.h"

namespace rtk
{

void RegisterIOFactories()
{
  // First unregister GDCMImageIO to let ImageXDCM
  std::list< itk::ObjectFactoryBase * > fl = itk::GDCMImageIOFactory::GetRegisteredFactories();
  for (std::list< itk::ObjectFactoryBase * >::iterator it = fl.begin(); it != fl.end(); ++it)
    if (dynamic_cast<itk::GDCMImageIOFactory *>(*it))
    {
    itk::GDCMImageIOFactory::UnRegisterFactory(*it);
    }
  rtk::HndImageIOFactory::RegisterOneFactory();
  rtk::XimImageIOFactory::RegisterOneFactory();
  rtk::HisImageIOFactory::RegisterOneFactory();
  rtk::ImagXImageIOFactory::RegisterOneFactory();
  rtk::DCMImagXImageIOFactory::RegisterOneFactory();
  rtk::EdfImageIOFactory::RegisterOneFactory();
  rtk::XRadImageIOFactory::RegisterOneFactory();
  rtk::OraImageIOFactory::RegisterOneFactory();
  itk::GDCMImageIOFactory::RegisterOneFactory();
}

} //namespace rtk
