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

#ifndef rtkOraImageIO_h
#define rtkOraImageIO_h

// This is done to avoid any interference with zlib
#ifdef OF
# undef OF
#endif

#include <itkMetaImageIO.h>

#include "rtkMacro.h"

namespace rtk
{

/** \class OraImageIO
 * \brief Class for reading Ora Image file format
 *
 * The ora image file format is used by medPhoton, extension of the header file
 * is ora.xml which points to a mhd and a raw files.
 *
 * \author Simon Rit
 *
 * \ingroup IOFilters
 */
class OraImageIO : public itk::MetaImageIO
{
public:
  /** Standard class typedefs. */
  typedef OraImageIO              Self;
  typedef itk::MetaImageIO        Superclass;
  typedef itk::SmartPointer<Self> Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(OraImageIO, itk::MetaImageIO);

  /*-------- This part of the interface deals with reading data. ------ */

  /** Determine the file type. Returns true if this ImageIO can read the
   * file specified. */
  virtual bool CanReadFile( const char* FileNameToRead ) ITK_OVERRIDE;

  /** Set the spacing and dimension information for the set filename. */
  virtual void ReadImageInformation() ITK_OVERRIDE;

  /** Reads the data from disk into the memory buffer provided. */
  virtual void Read(void *buffer) ITK_OVERRIDE;

  bool CanWriteFile(const char* filename) ITK_OVERRIDE;

protected:
  std::string m_MetaFileName;
}; // end class OraImageIO
}

#endif /* end #define rtkOraImageIO_h */
