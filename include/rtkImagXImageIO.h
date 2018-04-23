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

#ifndef rtkImagXImageIO_h
#define rtkImagXImageIO_h

#include <itkImageIOBase.h>
#include <fstream>
#include <string.h>

#include "rtkMacro.h"

namespace rtk
{

/** \class ImagXImageIO
 *
 * TODO
 *
 */
class ImagXImageIO : public itk::ImageIOBase
{
public:
  /** Standard class typedefs. */
  typedef ImagXImageIO            Self;
  typedef itk::ImageIOBase        Superclass;
  typedef itk::SmartPointer<Self> Pointer;

  ImagXImageIO() : Superclass() {}

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImagXImageIO, ImageIOBase);

  /*-------- This part of the interface deals with reading data. ------ */
  void ReadImageInformation() ITK_OVERRIDE;

  bool CanReadFile( const char* FileNameToRead ) ITK_OVERRIDE;

  void Read(void * buffer) ITK_OVERRIDE;

  /*-------- This part of the interfaces deals with writing data. ----- */
  virtual void WriteImageInformation(bool keepOfStream);

  void WriteImageInformation() ITK_OVERRIDE { WriteImageInformation(false); }
  bool CanWriteFile(const char* filename) ITK_OVERRIDE;

  void Write(const void* buffer) ITK_OVERRIDE;

protected:
  std::string m_RawFileName;
};

} // end namespace

#endif
