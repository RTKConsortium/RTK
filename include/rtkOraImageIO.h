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

#ifndef rtkOraImageIO_h
#define rtkOraImageIO_h

// This is done to avoid any interference with zlib
#ifdef OF
#  undef OF
#endif

#include <itkMetaImageIO.h>

#include "RTKExport.h"
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
 * \ingroup RTK IOFilters
 */
class RTK_EXPORT OraImageIO : public itk::MetaImageIO
{
public:
  /** Standard class type alias. */
  using Self = OraImageIO;
  using Superclass = itk::MetaImageIO;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(OraImageIO);

  /*-------- This part of the interface deals with reading data. ------ */

  /** Determine the file type. Returns true if this ImageIO can read the
   * file specified. */
  bool
  CanReadFile(const char * FileNameToRead) override;

  /** Set the spacing and dimension information for the set filename. */
  void
  ReadImageInformation() override;

  /** Reads the data from disk into the memory buffer provided. */
  void
  Read(void * buffer) override;

  bool
  CanWriteFile(const char * filename) override;

protected:
  std::string m_MetaFileName;
}; // end class OraImageIO
} // namespace rtk

#endif /* end #define rtkOraImageIO_h */
