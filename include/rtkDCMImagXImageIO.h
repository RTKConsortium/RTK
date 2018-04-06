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

#ifndef rtkDCMImagXImageIO_h
#define rtkDCMImagXImageIO_h

#include <itkGDCMImageIO.h>

#include "rtkMacro.h"

namespace rtk
{

/** \class DCMImagXImageIO
 *
 *
 *
 */
class DCMImagXImageIO : public itk::GDCMImageIO
{
public:
  /** Standard class typedefs. */
  typedef DCMImagXImageIO         Self;
  typedef itk::GDCMImageIO        Superclass;
  typedef itk::SmartPointer<Self> Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DCMImagXImageIO, itk::GDCMImageIO);

  void ReadImageInformation() ITK_OVERRIDE;
  bool CanReadFile( const char* FileNameToRead ) ITK_OVERRIDE;
  bool CanWriteFile(const char* filename) ITK_OVERRIDE;

protected:
  DCMImagXImageIO() {}
  ~DCMImagXImageIO() {}
};

} // end namespace

#endif
