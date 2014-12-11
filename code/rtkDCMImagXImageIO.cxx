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

#include "rtkDCMImagXImageIO.h"

#include <gdcmImageReader.h>
#include <gdcmAttribute.h>

void rtk::DCMImagXImageIO::ReadImageInformation()
{
  Superclass::ReadImageInformation();

  this->SetOrigin(0, (this->GetDimensions(0)-1)*(-0.5)*this->GetSpacing(0) );
  this->SetOrigin(1, (this->GetDimensions(1)-1)*(-0.5)*this->GetSpacing(1) );
  this->SetOrigin(2, 0.);
}

bool rtk::DCMImagXImageIO::CanReadFile(const char* FileNameToRead)
{
  if(!Superclass::CanReadFile(FileNameToRead))
    return false;

  // Check IBA label, reading manufacturer's name
  gdcm::ImageReader reader;
  reader.SetFileName(FileNameToRead);
  reader.Read();

  gdcm::DataSet &ds = reader.GetFile().GetDataSet();
  gdcm::Attribute<0x8,0x70> at1;
  at1.SetFromDataSet(ds);
  std::string value = at1.GetValue();

  return bool(value=="IBA ");
}

bool rtk::DCMImagXImageIO::CanWriteFile( const char* itkNotUsed(FileNameToWrite) )
{
  return false;
}
