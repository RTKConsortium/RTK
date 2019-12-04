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

#ifndef rtkHisImageIO_h
#define rtkHisImageIO_h

// itk include
#include <itkImageIOBase.h>
#include "rtkMacro.h"

namespace rtk
{

/** \class HisImageIO
 * \brief Class for reading His Image file format
 *
 * The his image file format is used by Perkin Elmer flat panels.
 *
 * \author Simon Rit
 *
 * \ingroup RTK IOFilters
 */
class HisImageIO : public itk::ImageIOBase
{
public:
  /** Standard class type alias. */
  using Self = HisImageIO;
  using Superclass = itk::ImageIOBase;
  using Pointer = itk::SmartPointer<Self>;
  using PixelType = signed short int;

  HisImageIO()
    : Superclass()
  {
    ;
  }

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(HisImageIO, itk::ImageIOBase);

  /*-------- This part of the interface deals with reading data. ------ */
  void
  ReadImageInformation() override;

  bool
  CanReadFile(const char * FileNameToRead) override;

  void
  Read(void * buffer) override;

  /*-------- This part of the interfaces deals with writing data. ----- */
  virtual void
  WriteImageInformation(bool /*keepOfStream*/)
  {
    ;
  }

  void
  WriteImageInformation() override
  {
    WriteImageInformation(false);
  }

  bool
  CanWriteFile(const char * filename) override;

  void
  Write(const void * buffer) override;

protected:
  int m_HeaderSize;

}; // end class HisImageIO
} // namespace rtk

#endif /* end #define rtkHisImageIO_h */
