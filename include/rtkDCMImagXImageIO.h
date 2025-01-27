/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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

#include "RTKExport.h"
#include "rtkMacro.h"

namespace rtk
{

/** \class DCMImagXImageIO
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
class RTK_EXPORT DCMImagXImageIO : public itk::GDCMImageIO
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DCMImagXImageIO);

  /** Standard class type alias. */
  using Self = DCMImagXImageIO;
  using Superclass = itk::GDCMImageIO;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(DCMImagXImageIO);

  void
  ReadImageInformation() override;
  bool
  CanReadFile(const char * FileNameToRead) override;
  bool
  CanWriteFile(const char * filename) override;

protected:
  DCMImagXImageIO() = default;
  ~DCMImagXImageIO() override = default;
};

} // namespace rtk

#endif
