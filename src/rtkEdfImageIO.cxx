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

#include "rtkEdfImageIO.h"

// itk include (for itkReadRawBytesAfterSwappingMacro)
#include <itkRawImageIO.h>
#include <itk_zlib.h>

//--------------------------------------------------------------------
/* Find value_ptr as pointer to the parameter of the given key in the header.
 * Returns NULL on success.
 */
char*
rtk::EdfImageIO::edf_findInHeader( char* header, const char* key )
{
  char *value_ptr = strstr( header, key );

  if (!value_ptr) return ITK_NULLPTR;
  /* an edf line is "key     = value ;" */
  value_ptr = 1 + strchr( value_ptr + strlen(key), '=' );
  while (isspace(*value_ptr) ) value_ptr++;
  return value_ptr;
}

//--------------------------------------------------------------------
// Read Image Information
void rtk::EdfImageIO::ReadImageInformation()
{
  int    k;
  char * header = ITK_NULLPTR;
  int    header_size = 0;
  char * p;
  gzFile inp;

  inp = gzopen(m_FileName.c_str(), "rb");
  if (!inp)
    itkGenericExceptionMacro(<< "Cannot open input file " << m_FileName);

  // read header: it is a multiple of 512 B ending by "}\n"
  while (header_size == 0 || strncmp(&header[header_size-2],"}\n",2) ) {
    int header_size_prev = header_size;
    header_size += 512;
    if (!header)
      header = (char*)malloc(header_size+1);
    else
      header = (char*)realloc(header, header_size+1);
    header[header_size_prev] = 0; /* protection against empty file */
    // fread(header+header_size_prev, 512, 1, fp);
    k = gzread(inp, header+header_size_prev, 512);
    if (k < 512) { /* protection against infinite loop */
      gzclose(inp);
      free(header);
      itkGenericExceptionMacro(<< "Damaged EDF header of "
                               << m_FileName
                               << ": not multiple of 512 B.");
      }
    header[header_size] = 0; /* end of string: protection against strstr later
                               on */
    }

  // parse the header
  int   dim1 = -1, dim2 = -1, datalen = -1;
  char *otherfile_name = ITK_NULLPTR; // this file, or another file with the data (EDF vs
                            // EHF formats)
  int   otherfile_skip = 0;

  if ( (p = edf_findInHeader(header, "EDF_BinaryFileName") ) ) {
    int plen = strcspn(p, " ;\n");
    otherfile_name = (char*)realloc(otherfile_name, plen+1);
    strncpy(otherfile_name, p, plen);
    otherfile_name[plen] = '\0';
    if ( (p = edf_findInHeader(header, "EDF_BinaryFilePosition") ) )
      otherfile_skip = atoi(p);
    }

  if ( (p = edf_findInHeader(header, "Dim_1") ) )
    dim1 = atoi(p);
  if ( (p = edf_findInHeader(header, "Dim_2") ) )
    dim2 = atoi(p);

//  int orig1 = -1, orig2 = -1;
//  if ((p = edf_findInHeader(header, "row_beg")))
//    orig1 = atoi(p);
//  if ((p = edf_findInHeader(header, "col_beg")))
//    orig2 = atoi(p);

  static const struct table3 edf_datatype_table[] =
    {
          { "UnsignedByte",    U_CHAR_DATATYPE,  1 },
          { "SignedByte",      CHAR_DATATYPE,    1 },
          { "UnsignedShort",   U_SHORT_DATATYPE, 2 },
          { "SignedShort",     SHORT_DATATYPE,   2 },
          { "UnsignedInteger", U_INT_DATATYPE,   4 },
          { "SignedInteger",   INT_DATATYPE,     4 },
          { "UnsignedLong",    U_L_INT_DATATYPE, 4 },
          { "SignedLong",      L_INT_DATATYPE,   4 },
          { "FloatValue",      FLOAT_DATATYPE,   4 },
          { "DoubleValue",     DOUBLE_DATATYPE,  8 },
          { "Float",           FLOAT_DATATYPE,   4 }, // Float and FloatValue
                                                      // are synonyms
          { "Double",          DOUBLE_DATATYPE,  8 }, // Double and DoubleValue
                                                      // are synonyms
          { ITK_NULLPTR, -1, -1 }
    };
  if ( (p = edf_findInHeader(header, "DataType") ) ) {
    k = lookup_table3_nth(edf_datatype_table, p);
    if (k < 0) { // unknown EDF DataType
      gzclose(inp);
      free(header);
      itkGenericExceptionMacro( <<"Unknown EDF datatype \""
                                << p
                                << "\"");
      }
    datalen = edf_datatype_table[k].sajzof;
    switch(edf_datatype_table[k].value) {
      case U_CHAR_DATATYPE:
        SetComponentType(itk::ImageIOBase::UCHAR);
        break;
      case CHAR_DATATYPE:
        SetComponentType(itk::ImageIOBase::CHAR);
        break;
      case U_SHORT_DATATYPE:
        SetComponentType(itk::ImageIOBase::USHORT);
        break;
      case SHORT_DATATYPE:
        SetComponentType(itk::ImageIOBase::SHORT);
        break;
      case U_INT_DATATYPE:
        SetComponentType(itk::ImageIOBase::UINT);
        break;
      case INT_DATATYPE:
        SetComponentType(itk::ImageIOBase::INT);
        break;
      case U_L_INT_DATATYPE:
        SetComponentType(itk::ImageIOBase::UINT);
        break;
      case L_INT_DATATYPE:
        SetComponentType(itk::ImageIOBase::INT);
        break;
      case FLOAT_DATATYPE:
        SetComponentType(itk::ImageIOBase::FLOAT);
        break;
      case DOUBLE_DATATYPE:
        SetComponentType(itk::ImageIOBase::DOUBLE);
        break;
      }
    }

  static const struct table edf_byteorder_table[] =
    {
          { "LowByteFirst",  LittleEndian }, /* little endian */
          { "HighByteFirst", BigEndian },    /* big endian */
          { ITK_NULLPTR, -1 }
    };

  int byteorder = LittleEndian;
  if ( (p = edf_findInHeader(header, "ByteOrder") ) ) {
    k = lookup_table_nth(edf_byteorder_table, p);
    if (k >= 0) {

      byteorder = edf_byteorder_table[k].value;
      if(byteorder==LittleEndian)
        this->SetByteOrder(LittleEndian);
      else
        this->SetByteOrder(BigEndian);
      }
    } else
    itkWarningMacro(<<"ByteOrder not specified in the header! Not swapping bytes (figure may not be correct).");
  // Get and verify size of the data:
  int datasize = dim1 * dim2 * datalen;
  if ( (p = edf_findInHeader(header, "Size") ) ) {
    int d = atoi(p);
    if (d != datasize) {
      itkWarningMacro(<< "Size " << datasize << " is not "
                      << dim1 << 'x' << dim2 << "x" << datalen
                      << " = " << d << ". Supposing the latter.");
      }
    }

  // EHF files: binary data are in another file than the header file
  m_BinaryFileName = m_FileName;
  m_BinaryFileSkip = header_size;
  if (otherfile_name) {
    m_BinaryFileName = std::string(otherfile_name);
    m_BinaryFileSkip = otherfile_skip;
    }

  double spacing = 1.;
  if ( (p = edf_findInHeader(header, "optic_used") ) )
    {
    spacing = atof(p);
    if(spacing == 0.)
      spacing = 1.;
    }


  free(header);
  gzclose(inp);

  SetNumberOfDimensions(2);
  SetDimensions(0, dim1);
  SetDimensions(1, dim2);
  SetSpacing(0, spacing);
  SetSpacing(1, spacing);
  SetOrigin(0, 0.);
  SetOrigin(1, 0.);
} ////

//--------------------------------------------------------------------
// Read Image Information
bool rtk::EdfImageIO::CanReadFile(const char* FileNameToRead)
{
  std::string                  filename(FileNameToRead);
  const std::string::size_type it = filename.find_last_of( "." );
  std::string                  fileExt( filename, it+1, filename.length() );

  if (fileExt != std::string("edf") ) return false;
  return true;
} ////

//--------------------------------------------------------------------
// Read Image Content
void rtk::EdfImageIO::Read(void * buffer)
{
  gzFile inp;

  inp = gzopen(m_BinaryFileName.c_str(), "rb");
  if (!inp)
    itkGenericExceptionMacro(<< "Cannot open file \"" << m_FileName << "\"");
  gzseek(inp, m_BinaryFileSkip, SEEK_SET);

  // read the data (image)
  long numberOfBytesToBeRead = GetComponentSize();
  for(unsigned int i=0; i<GetNumberOfDimensions(); i++) numberOfBytesToBeRead *= GetDimensions(i);

  if (numberOfBytesToBeRead != gzread(inp, buffer, numberOfBytesToBeRead) )
    itkGenericExceptionMacro(<< "The image " << m_BinaryFileName << " cannot be read completely.");

  gzclose(inp);

  // Adapted from itkRawImageIO
    {
    using namespace itk;
    // Swap bytes if necessary
    if itkReadRawBytesAfterSwappingMacro( unsigned short, USHORT )
    else if itkReadRawBytesAfterSwappingMacro( short, SHORT )
    else if itkReadRawBytesAfterSwappingMacro( char, CHAR )
    else if itkReadRawBytesAfterSwappingMacro( unsigned char, UCHAR )
    else if itkReadRawBytesAfterSwappingMacro( unsigned int, UINT )
    else if itkReadRawBytesAfterSwappingMacro( int, INT )
    else if itkReadRawBytesAfterSwappingMacro( unsigned int, UINT )
    else if itkReadRawBytesAfterSwappingMacro( int, INT )
    else if itkReadRawBytesAfterSwappingMacro( float, FLOAT )
    else if itkReadRawBytesAfterSwappingMacro( double, DOUBLE );
    }
}

//--------------------------------------------------------------------
// Write Image Information
void rtk::EdfImageIO::WriteImageInformation( bool itkNotUsed(keepOfStream) )
{
}

//--------------------------------------------------------------------
// Write Image Information
bool rtk::EdfImageIO::CanWriteFile( const char* itkNotUsed(FileNameToWrite) )
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void rtk::EdfImageIO::Write( const void * itkNotUsed(buffer) )
{
} ////
