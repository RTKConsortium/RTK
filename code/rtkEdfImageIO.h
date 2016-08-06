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

#ifndef rtkEdfImageIO_h
#define rtkEdfImageIO_h

#include <itkImageIOBase.h>
#include <fstream>
#include <string.h>

#include "rtkMacro.h"

namespace rtk {

/** \class EdfImageIO
 * \brief Class for reading Edf image file format. Edf is the format of
 * X-ray projection images at the ESRF.
 *
 * \author Simon Rit
 *
 * \ingroup IOFilters
 */
class EdfImageIO : public itk::ImageIOBase
{
public:
  /** Standard class typedefs. */
  typedef EdfImageIO              Self;
  typedef itk::ImageIOBase        Superclass;
  typedef itk::SmartPointer<Self> Pointer;

  EdfImageIO() : Superclass() {
  }

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(EdfImageIO, ImageIOBase);

  /*-------- This part of the interface deals with reading data. ------ */
  void ReadImageInformation() ITK_OVERRIDE;

  bool CanReadFile( const char* FileNameToRead ) ITK_OVERRIDE;

  void Read(void * buffer) ITK_OVERRIDE;

  /*-------- This part of the interfaces deals with writing data. ----- */
  virtual void WriteImageInformation(bool keepOfStream);

  void WriteImageInformation() ITK_OVERRIDE {
    WriteImageInformation(false);
  }

  bool CanWriteFile(const char* filename) ITK_OVERRIDE;

  void Write(const void* buffer) ITK_OVERRIDE;

protected:
  std::string m_BinaryFileName;
  int         m_BinaryFileSkip;

  static char* edf_findInHeader( char* header, const char* key );

  /* List of EDF supported datatypes
   */
  enum DataType {
    U_CHAR_DATATYPE = 0, CHAR_DATATYPE,        //  8 bits = 1 B
    U_SHORT_DATATYPE,    SHORT_DATATYPE,       // 16 bits = 2 B
    U_INT_DATATYPE,      INT_DATATYPE,         // 32 bits = 4 B
    U_L_INT_DATATYPE,    L_INT_DATATYPE,       // 32 bits = 4 B
    FLOAT_DATATYPE,      DOUBLE_DATATYPE,      // 4 B, 8 B
    UNKNOWN_DATATYPE = -1
    };

  /* Note - compatibility:
    Unsigned8 = 1,Signed8,  Unsigned16, Signed16,
    Unsigned32,   Signed32, Unsigned64, Signed64,
    FloatIEEE32,  DoubleIEEE64
  */

  /***************************************************************************
   * Tables
   ***************************************************************************/

  // table key-value structure
  struct table {
    const char *key;
    int value;
    };

  struct table3 {
    const char *key;
    int value;
    short sajzof;
    };

  /* Returns index of the table tbl whose key matches the beginning of the
   * search string search_str.
   * It returns index into the table or -1 if there is no match.
   */
  static int
  lookup_table_nth( const struct table *tbl, const char *search_str )
  {
    int k = -1;

    while (tbl[++k].key)
      if (tbl[k].key && !strncmp(search_str, tbl[k].key, strlen(tbl[k].key) ) )
        return k;
    return -1; // not found
  }

  static int
  lookup_table3_nth( const struct table3 *tbl, const char *search_str )
  {
    int k = -1;

    while (tbl[++k].key)
      if (tbl[k].key && !strncmp(search_str, tbl[k].key, strlen(tbl[k].key) ) )
        return k;
    return -1; // not found
  }

  ///* Orientation of axes of the raster, as the binary matrix is saved in
  // * the file. (Determines the scanning direction, or the "fastest" index
  // * of the matrix in the data file.)
  // */
  //enum EdfRasterAxes {
  //RASTER_AXES_XrightYdown, // matricial format: rows, columns
  //RASTER_AXES_XrightYup    // cartesian coordinate system
  //    // other 6 combinations not available (not needed until now)
  //};

  //static const struct table rasteraxes_table[] =
  //{
  //    { "XrightYdown", RASTER_AXES_XrightYdown },
  //    { "XrightYup",   RASTER_AXES_XrightYup },
  //    { NULL, -1 }
  //};

}; // end class EdfImageIO

} // end namespace

#endif
