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

// std include
#include <cstdio>
#include <valarray>
#include <numeric>

#include "rtkXimImageIO.h"
#include <itkMetaDataObject.h>

#define PROPERTY_NAME_MAX_LENGTH 256

template <typename T>
size_t
rtk::XimImageIO::SetPropertyValue(char * property_name, Int4 value_length, FILE * fp, Xim_header * xim)
{
  T      property_value;
  T *    unused_property_value = nullptr;
  size_t addNelements = 0;

  if (value_length > 1)
  {
    unused_property_value = new T[value_length];
    addNelements += fread((void *)unused_property_value, sizeof(T), value_length, fp);
    delete[] unused_property_value;
    return addNelements;
  }

  addNelements += fread((void *)&property_value, sizeof(T), value_length, fp);

  if (strncmp(property_name, "CouchLat", 8) == 0)
    xim->dCouchLat = property_value;
  else if (strncmp(property_name, "CouchLng", 8) == 0)
    xim->dCouchLng = property_value;
  else if (strncmp(property_name, "CouchVrt", 8) == 0)
    xim->dCouchVrt = property_value;
  else if (strncmp(property_name, "DataOffset", 10) == 0)
    xim->nPixelOffset = property_value;
  else if (strncmp(property_name, "KVSourceRtn", 11) == 0)
    xim->dCTProjectionAngle = property_value;
  else if (strncmp(property_name, "KVDetectorLat", 13) == 0)
    xim->dDetectorOffsetX = property_value;
  else if (strncmp(property_name, "KVDetectorLng", 13) == 0)
    xim->dDetectorOffsetY = property_value;
  else if (strncmp(property_name, "KVCollimatorX1", 14) == 0)
    xim->dCollX1 = property_value;
  else if (strncmp(property_name, "KVCollimatorX2", 14) == 0)
    xim->dCollX2 = property_value;
  else if (strncmp(property_name, "KVCollimatorY1", 14) == 0)
    xim->dCollY1 = property_value;
  else if (strncmp(property_name, "KVCollimatorY2", 14) == 0)
    xim->dCollY2 = property_value;
  else if (strncmp(property_name, "KVKiloVolts", 11) == 0)
    xim->dXRayKV = property_value;
  else if (strncmp(property_name, "KVMilliAmperes", 14) == 0)
    xim->dXRayMA = property_value;
  else if (strncmp(property_name, "KVNormChamber", 13) == 0)
    xim->dCTNormChamber = property_value;
  else if (strncmp(property_name, "MMTrackingRemainderX", 20) == 0)
    xim->dGating4DInfoX = property_value;
  else if (strncmp(property_name, "MMTrackingRemainderY", 20) == 0)
    xim->dGating4DInfoY = property_value;
  else if (strncmp(property_name, "MMTrackingRemainderZ", 20) == 0)
    xim->dGating4DInfoZ = property_value;
  else if (strncmp(property_name, "MVCollimatorRtn", 15) == 0)
    xim->dCollRtn = property_value;
  else if (strncmp(property_name, "MVCollimatorX1", 14) == 0)
    xim->dCollX1 = property_value;
  else if (strncmp(property_name, "MVCollimatorX2", 14) == 0)
    xim->dCollX2 = property_value;
  else if (strncmp(property_name, "MVCollimatorY1", 14) == 0)
    xim->dCollY1 = property_value;
  else if (strncmp(property_name, "MVCollimatorY2", 14) == 0)
    xim->dCollY2 = property_value;
  else if (strncmp(property_name, "MVDoseRate", 10) == 0)
    xim->dDoseRate = property_value;
  else if (strncmp(property_name, "MVEnergy", 8) == 0)
    xim->dEnergy = property_value;
  else if (strncmp(property_name, "PixelHeight", 11) == 0)
    xim->dIDUResolutionY = property_value * 10.0;
  else if (strncmp(property_name, "PixelWidth", 10) == 0)
    xim->dIDUResolutionX = property_value * 10.0;
  return addNelements;
}

//--------------------------------------------------------------------
// Read Image Information
void
rtk::XimImageIO::ReadImageInformation()
{
  Xim_header xim;
  FILE *     fp = nullptr;

  fp = fopen(m_FileName.c_str(), "rb");
  if (fp == nullptr)
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);
  size_t nelements = 0;
  nelements += fread((void *)xim.sFileType, sizeof(char), 8, fp);
  nelements += fread((void *)&xim.FileVersion, sizeof(Int4), 1, fp);
  nelements += fread((void *)&xim.SizeX, sizeof(Int4), 1, fp);
  nelements += fread((void *)&xim.SizeY, sizeof(Int4), 1, fp);

  nelements += fread((void *)&xim.dBitsPerPixel, sizeof(Int4), 1, fp);
  nelements += fread((void *)&xim.dBytesPerPixel, sizeof(Int4), 1, fp);
  m_BytesPerPixel = xim.dBytesPerPixel;
  nelements += fread((void *)&xim.dCompressionIndicator, sizeof(Int4), 1, fp);
  m_ImageDataStart = ftell(fp);
  if (xim.dCompressionIndicator == 1)
  {
    nelements += fread((void *)&xim.lookUpTableSize, sizeof(Int4), 1, fp);
    fseek(fp, xim.lookUpTableSize, SEEK_CUR);
    nelements += fread((void *)&xim.compressedPixelBufferSize, sizeof(Int4), 1, fp);
    fseek(fp, xim.compressedPixelBufferSize, SEEK_CUR);
    nelements += fread((void *)&xim.unCompressedPixelBufferSize, sizeof(Int4), 1, fp);
    if (nelements != /*char*/ 8 + /*Int4*/ 9)
      itkGenericExceptionMacro(<< "Could not read header data in " << m_FileName);
  }
  else
  {
    nelements += fread((void *)&xim.unCompressedPixelBufferSize, sizeof(Int4), 1, fp);
    fseek(fp, xim.unCompressedPixelBufferSize, SEEK_CUR);
    if (nelements != /*char*/ 8 + /*Int4*/ 7)
      itkGenericExceptionMacro(<< "Could not read header data in " << m_FileName);
  }

  // Histogram Reading:
  nelements += fread((void *)&xim.binsInHistogram, sizeof(Int4), 1, fp);
  fseek(fp, xim.binsInHistogram * sizeof(Int4), SEEK_CUR);
  /* // Replace the two lines above with this if you actually want the histogram:
  int nhistElements = 0;
  nhistElements += fread((void *)&xim.binsInHistogram, sizeof(Int4), 1, fp);

  auto xim.histogramData = (int*) malloc(sizeof(int) * xim.binsInHistogram);

  nhistElements += fread((void *)&xim.histogramData, sizeof(Int4), xim.binsInHistogram, fp);
  if (nhistElements != (xim.binsInHistogram + 1))
  {
    itkGenericExceptionMacro(<< "Could not read histogram from header data in " << m_FileName);
  }
  // free(xim.histogramData); // <- Remember to put this after you are done with the histogram
  */

  // Properties Readding:
  nelements += fread((void *)&xim.numberOfProperties, sizeof(Int4), 1, fp);
  Int4   property_name_length = 0;
  char   property_name[PROPERTY_NAME_MAX_LENGTH];
  Int4   property_type = 0;
  Int4   property_value_length = 0;
  size_t theoretical_nelements = nelements; // Same as reseting

  for (Int4 i = 0; i < xim.numberOfProperties; i++)
  {
    nelements += fread((void *)&property_name_length, sizeof(Int4), 1, fp);
    if (property_name_length > PROPERTY_NAME_MAX_LENGTH)
      itkGenericExceptionMacro(<< "Property name is too long, i.e., " << property_name_length);
    nelements += fread((void *)&property_name, sizeof(char), property_name_length, fp);
    nelements += fread((void *)&property_type, sizeof(Int4), 1, fp);
    theoretical_nelements += property_name_length + 2;

    switch (property_type)
    {
      case 0: // property_value type = uint32
        nelements += SetPropertyValue<Int4>(property_name, 1, fp, &xim);
        theoretical_nelements++;
        break;
      case 1: // property_value type = double
        theoretical_nelements++;
        nelements += SetPropertyValue<double>(property_name, 1, fp, &xim);
        break;
      case 2: // property_value type = length * char
        nelements += fread((void *)&property_value_length, sizeof(Int4), 1, fp);
        theoretical_nelements += property_value_length + 1;
        nelements += SetPropertyValue<char>(property_name, property_value_length, fp, &xim);
        break;
      case 4: // property_value type = length * double
        nelements += fread((void *)&property_value_length, sizeof(Int4), 1, fp);
        nelements += SetPropertyValue<double>(property_name, property_value_length / 8, fp, &xim);
        theoretical_nelements += property_value_length / 8 + 1;
        break;
      case 5: // property_value type = length * uint32
        nelements += fread((void *)&property_value_length, sizeof(Int4), 1, fp);
        nelements += SetPropertyValue<Int4>(property_name, property_value_length / 4, fp, &xim);
        theoretical_nelements += property_value_length / 4 + 1;
        break;
      default:
        std::cout << "\n\nProperty name: " << property_name << ", type: " << property_type
                  << ", is not supported! ABORTING decoding!";
        return;
    }
  }
  if (nelements != theoretical_nelements)
  {
    std::cout << nelements << " != " << theoretical_nelements << std::endl;
    itkGenericExceptionMacro(<< "Could not read properties of " << m_FileName);
  }
  if (fclose(fp) != 0)
    itkGenericExceptionMacro(<< "Could not close file: " << m_FileName);

  /* Convert xim to ITK image information */
  this->SetNumberOfDimensions(2);
  this->SetDimensions(0, xim.SizeX);
  this->SetDimensions(1, xim.SizeY);

  this->SetSpacing(0, xim.dIDUResolutionX); // set to PixelHeight/Width
  this->SetSpacing(1, xim.dIDUResolutionY);
  this->SetOrigin(0, -0.5 * (xim.SizeX - 1) * xim.dIDUResolutionX); // SR: assumed centered
  this->SetOrigin(1, -0.5 * (xim.SizeY - 1) * xim.dIDUResolutionY); // SR: assumed centered

  this->SetPixelType(itk::IOPixelEnum::SCALAR);
  this->SetComponentType(itk::ImageIOBase::IOComponentEnum::UINT); // 32 bit ints

  /* Store important meta information in the meta data dictionary */
  if (xim.SizeX * xim.SizeY != 0)
  {
    itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTProjectionAngle", xim.dCTProjectionAngle);
    itk::EncapsulateMetaData<double>(
      this->GetMetaDataDictionary(), "dDetectorOffsetX", xim.dDetectorOffsetX * 10.0); // cm->mm Lat
    itk::EncapsulateMetaData<double>(
      this->GetMetaDataDictionary(), "dDetectorOffsetY", xim.dDetectorOffsetY * 10.0); // cm->mm Lng
  }
  else
  {
    itk::ImageIORegion ioreg;
    ioreg.SetIndex(0, 0);
    ioreg.SetIndex(1, 0);
    ioreg.SetSize(0, 0);
    ioreg.SetSize(1, 0);
    this->SetIORegion(ioreg);
    unsigned int imdim[] = { 0, 0 };
    this->Resize(2, imdim);
    itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTProjectionAngle", 6000);
  }
}
//--------------------------------------------------------------------
bool
rtk::XimImageIO::CanReadFile(const char * FileNameToRead)
{
  std::string                  filename(FileNameToRead);
  const std::string::size_type it = filename.find_last_of('.');
  std::string                  fileExt(filename, it + 1, filename.length());

  if (fileExt != std::string("xim"))
    return false;

  FILE * fp = nullptr;
  fp = fopen(filename.c_str(), "rb");
  if (fp == nullptr)
  {
    std::cerr << "Could not open file (for reading): " << m_FileName << std::endl;
    return false;
  }

  size_t nelements = 0;
  char   sfiletype[8];
  Int4   fileversion = 0, sizex = 0, sizey = 0;

  nelements += fread((void *)&sfiletype[0], sizeof(char), 8, fp);
  nelements += fread((void *)&fileversion, sizeof(Int4), 1, fp);
  nelements += fread((void *)&sizex, sizeof(Int4), 1, fp);
  nelements += fread((void *)&sizey, sizeof(Int4), 1, fp);

  if (nelements != 8 + 3)
  {
    std::cerr << "Could not read initial header data in " << m_FileName << std::endl;
    fclose(fp);
    return false;
  }
  if (sizex * sizey <= 0)
  {
    std::cerr << "Imagedata was of size (x, y)=(" << sizex << ", " << sizey << ") in " << filename << std::endl;
    fclose(fp);
    return false;
  }

  if (fclose(fp) != 0)
  {
    std::cerr << "Could not close file (after reading): " << m_FileName << std::endl;
    return false;
  }

  return true;
}


//--------------------------------------------------------------------
template <typename T>
inline T
cast_binary_char_to(const unsigned char * bin_vals, const size_t n_bytes)
{
  T out_val = 0;
  switch (n_bytes)
  {
    case 1:
      out_val = static_cast<T>(*(int8_t *)(void *)bin_vals);
      break;
    case 2:
      out_val = static_cast<T>(*(int16_t *)(void *)bin_vals);
      break;
    case 4:
      out_val = static_cast<T>(*(int32_t *)(void *)bin_vals);
      break;
  }
  return out_val;
}

inline size_t
lut_to_bytes(const char val)
{
  switch (val)
  {
    case 0:
      return 1;
    case 1:
      return 2;
    case 2:
      return 4;
    default: // only 0, 1 & 2 should be possible
      return 8;
  }
}

// Read Image Content
void
rtk::XimImageIO::Read(void * buffer)
{
  FILE * fp = nullptr;
  // Long is only garanteed to be AT LEAST 32 bits, it could be 64 bit
  Int4 * buf = (Int4 *)buffer;

  fp = fopen(m_FileName.c_str(), "rb");
  if (fp == nullptr)
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);

  if (fseek(fp, m_ImageDataStart, SEEK_SET) != 0)
    itkGenericExceptionMacro(<< "Could not seek to image data in: " << m_FileName);

  Int4 lookUpTableSize = 0;
  // De"compress" image
  if (1 != fread((void *)&lookUpTableSize, sizeof(Int4), 1, fp))
  {
    itkGenericExceptionMacro(<< "Could not read LUT size from: " << m_FileName);
  }
  auto m_lookup_table = std::valarray<unsigned char>(lookUpTableSize);
  if (lookUpTableSize != (Int4)fread((void *)&m_lookup_table[0], sizeof(unsigned char), lookUpTableSize, fp))
  {
    itkGenericExceptionMacro(<< "Could not read lookup table from Xim file: " << m_FileName);
  }

  Int4 compressedPixelBufferSize = 0;
  if (1 != fread((void *)&compressedPixelBufferSize, sizeof(Int4), 1, fp))
  {
    itkGenericExceptionMacro(<< "Could not get compressed pixel buffer size from: " << m_FileName);
  }

  const auto xdim = GetDimensions(0);
  const auto ydim = GetDimensions(1);
  if (xdim * ydim == 0)
  {
    itkGenericExceptionMacro(<< "Dimensions of image was 0 in: " << m_FileName);
  }

  if ((xdim + 1) != fread(&buf[0], sizeof(Int4), xdim + 1, fp))
    itkGenericExceptionMacro(<< "Could not read first row +1 in: " << m_FileName);

  auto byte_table_expr = m_lookup_table.apply([](const unsigned char & v) {
    unsigned char bytes = 0;
    bytes += lut_to_bytes(v & 0b00000011);        // 0x03
    bytes += lut_to_bytes((v & 0b00001100) >> 2); // 0x0C
    bytes += lut_to_bytes((v & 0b00110000) >> 4); // 0x30
    bytes += lut_to_bytes((v & 0b11000000) >> 6); // 0xC0
    return bytes;
  });

  std::valarray<unsigned char> byte_table(byte_table_expr);
  const auto                   total_bytes = std::accumulate(std::begin(byte_table), std::end(byte_table), 0ull);


  auto compr_img_buffer = std::valarray<unsigned char>(total_bytes);
  // total_bytes - 3 because the last two bits can be redundant (according to Xim docs)
  if ((total_bytes - 3) > fread((void *)&compr_img_buffer[0], sizeof(unsigned char), total_bytes, fp))
  {
    itkGenericExceptionMacro(<< "Could not read image buffer of Xim file: " << m_FileName);
  }

  size_t j = 0U;
  size_t i = xdim;
  size_t iminxdim = 0;

  for (auto lut_idx = 0U; lut_idx < static_cast<size_t>(lookUpTableSize); ++lut_idx)
  {
    const auto v = m_lookup_table[lut_idx];
    auto       bytes = lut_to_bytes(v & 0b00000011); // 0x03
    assert(bytes == 1 || bytes == 2 || bytes == 4);
    auto diff1 = cast_binary_char_to<Int4>(&compr_img_buffer[j], bytes);
    j += bytes;

    bytes = lut_to_bytes((v & 0b00001100) >> 2); // 0x0C
    assert(bytes == 1 || bytes == 2 || bytes == 4);
    auto diff2 = cast_binary_char_to<Int4>(&compr_img_buffer[j], bytes);
    j += bytes;

    bytes = lut_to_bytes((v & 0b00110000) >> 4); // 0x30
    assert(bytes == 1 || bytes == 2 || bytes == 4);
    auto diff3 = cast_binary_char_to<Int4>(&compr_img_buffer[j], bytes);
    j += bytes;

    bytes = lut_to_bytes((v & 0b11000000) >> 6); // 0xC0
    assert(bytes == 1 || bytes == 2 || bytes == 4);
    auto diff4 = cast_binary_char_to<Int4>(&compr_img_buffer[j], bytes);
    j += bytes;

    buf[i + 1] = diff1 + buf[i] + buf[iminxdim + 1] - buf[iminxdim];
    buf[i + 2] = diff2 + buf[i + 1] + buf[iminxdim + 2] - buf[iminxdim + 1];
    buf[i + 3] = diff3 + buf[i + 2] + buf[iminxdim + 3] - buf[iminxdim + 2];
    buf[i + 4] = diff4 + buf[i + 3] + buf[iminxdim + 4] - buf[iminxdim + 3];

    i += 4;
    iminxdim += 4;
  }

  assert(j == total_bytes);
  assert(i == (xdim * ydim));

  if (fclose(fp) != 0)
    itkGenericExceptionMacro(<< "Could not close file: " << m_FileName);
}

//--------------------------------------------------------------------
bool
rtk::XimImageIO::CanWriteFile(const char * itkNotUsed(FileNameToWrite))
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void
rtk::XimImageIO::Write(const void * itkNotUsed(buffer))
{
  // TODO?
}
