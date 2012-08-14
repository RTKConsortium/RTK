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

#ifndef __rtkHisImageIO_h
#define __rtkHisImageIO_h

// itk include
#include <itkImageIOBase.h>

namespace rtk
{

/** \class HisImageIO
 * \brief Class for reading His Image file format
 *
 * The his image file format is used by Perkin Elmer flat panels.
 *
 * \author Simon Rit
 *
 * \ingroup IOFilters
 */
class HisImageIO : public itk::ImageIOBase
{
public:
  /** Standard class typedefs. */
  typedef HisImageIO              Self;
  typedef itk::ImageIOBase        Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef signed short int        PixelType;

  HisImageIO() : Superclass() {
    ;
  }

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(HisImageIO, itk::ImageIOBase);

  /*-------- This part of the interface deals with reading data. ------ */
  virtual void ReadImageInformation();

  virtual bool CanReadFile( const char* FileNameToRead );

  virtual void Read(void * buffer);

  /*-------- This part of the interfaces deals with writing data. ----- */
  virtual void WriteImageInformation(bool keepOfStream) {
    ;
  }

  virtual void WriteImageInformation() {
    WriteImageInformation(false);
  }

  virtual bool CanWriteFile(const char* filename);

  virtual void Write(const void* buffer);

protected:
  int m_HeaderSize;

}; // end class HisImageIO
} // end namespace

#endif /* end #define __rtkHisImageIO_h */
