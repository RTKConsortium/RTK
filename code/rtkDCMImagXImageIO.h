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

#ifndef __rtkDCMImagXImageIO_h
#define __rtkDCMImagXImageIO_h

#include <itkGDCMImageIO.h>

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

  virtual void ReadImageInformation();
  virtual bool CanReadFile( const char* FileNameToRead );
  virtual bool CanWriteFile(const char* filename);

protected:
  DCMImagXImageIO() {}
  ~DCMImagXImageIO() {}
};

} // end namespace

#endif
