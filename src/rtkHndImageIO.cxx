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

#include "rtkHndImageIO.h"
#include <itkMetaDataObject.h>

//--------------------------------------------------------------------
// Read Image Information
void
rtk::HndImageIO::ReadImageInformation()
{
  Hnd_header hnd;
  FILE *     fp = nullptr;

  fp = fopen(m_FileName.c_str(), "rb");
  if (fp == nullptr)
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);

  size_t nelements = 0;
  nelements += fread((void *)hnd.sFileType, sizeof(char), 32, fp);
  nelements += fread((void *)&hnd.FileLength, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnd.sChecksumSpec, sizeof(char), 4, fp);
  nelements += fread((void *)&hnd.nCheckSum, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnd.sCreationDate, sizeof(char), 8, fp);
  nelements += fread((void *)hnd.sCreationTime, sizeof(char), 8, fp);
  nelements += fread((void *)hnd.sPatientID, sizeof(char), 16, fp);
  nelements += fread((void *)&hnd.nPatientSer, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnd.sSeriesID, sizeof(char), 16, fp);
  nelements += fread((void *)&hnd.nSeriesSer, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnd.sSliceID, sizeof(char), 16, fp);
  nelements += fread((void *)&hnd.nSliceSer, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnd.SizeX, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnd.SizeY, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnd.dSliceZPos, sizeof(double), 1, fp);
  nelements += fread((void *)hnd.sModality, sizeof(char), 16, fp);
  nelements += fread((void *)&hnd.nWindow, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnd.nLevel, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnd.nPixelOffset, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnd.sImageType, sizeof(char), 4, fp);
  nelements += fread((void *)&hnd.dGantryRtn, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dSAD, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dSFD, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCollX1, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCollX2, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCollY1, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCollY2, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCollRtn, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dFieldX, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dFieldY, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dBladeX1, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dBladeX2, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dBladeY1, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dBladeY2, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dIDUPosLng, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dIDUPosLat, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dIDUPosVrt, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dIDUPosRtn, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dPatientSupportAngle, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dTableTopEccentricAngle, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCouchVrt, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCouchLng, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCouchLat, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dIDUResolutionX, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dIDUResolutionY, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dImageResolutionX, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dImageResolutionY, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dEnergy, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dDoseRate, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dXRayKV, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dXRayMA, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dMetersetExposure, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dAcqAdjustment, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCTProjectionAngle, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dCTNormChamber, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dGatingTimeTag, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dGating4DInfoX, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dGating4DInfoY, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dGating4DInfoZ, sizeof(double), 1, fp);
  nelements += fread((void *)&hnd.dGating4DInfoTime, sizeof(double), 1, fp);

  if (nelements != /*char*/ 120 + /*itk::uint32_t*/ 10 + /*double*/ 41)
    itkGenericExceptionMacro(<< "Could not read header data in " << m_FileName);

  if (fclose(fp) != 0)
    itkGenericExceptionMacro(<< "Could not close file: " << m_FileName);

  /* Convert hnd to ITK image information */
  SetNumberOfDimensions(2);
  SetDimensions(0, hnd.SizeX);
  SetDimensions(1, hnd.SizeY);
  SetSpacing(0, hnd.dIDUResolutionX);
  SetSpacing(1, hnd.dIDUResolutionY);
  SetOrigin(0, -0.5 * (hnd.SizeX - 1) * hnd.dIDUResolutionX); // SR: assumed centered
  SetOrigin(1, -0.5 * (hnd.SizeY - 1) * hnd.dIDUResolutionY); // SR: assumed centered
  SetComponentType(itk::ImageIOBase::IOComponentEnum::UINT);

  /* Store important meta information in the meta data dictionary */
  itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTProjectionAngle", hnd.dCTProjectionAngle);
}

//--------------------------------------------------------------------
bool
rtk::HndImageIO::CanReadFile(const char * FileNameToRead)
{
  std::string                  filename(FileNameToRead);
  const std::string::size_type it = filename.find_last_of(".");
  std::string                  fileExt(filename, it + 1, filename.length());

  if (fileExt != std::string("hnd"))
    return false;
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
rtk::HndImageIO::Read(void * buffer)
{
  FILE * fp = nullptr;
  // Long is only garanteed to be AT LEAST 32 bits, it could be 64 bit
  using Int4 = itk::uint32_t;
  Int4 * buf = (Int4 *)buffer;

  fp = fopen(m_FileName.c_str(), "rb");
  if (fp == nullptr)
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);

  if (fseek(fp, 1024, SEEK_SET) != 0)
    itkGenericExceptionMacro(<< "Could not seek to image data in: " << m_FileName);

  const auto   xdim = GetDimensions(0);
  const auto   ydim = GetDimensions(1);
  const size_t lookUpTableSize = (ydim - 1) * xdim / 4;
  // De"compress" image
  auto m_lookup_table = std::valarray<unsigned char>(lookUpTableSize);
  if (lookUpTableSize != fread((void *)&m_lookup_table[0], sizeof(unsigned char), lookUpTableSize, fp))
  {
    itkGenericExceptionMacro(<< "Could not read lookup table from Hnd file: " << m_FileName);
  }

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
    itkGenericExceptionMacro(<< "Could not read image buffer of Hnd file: " << m_FileName);
  }

  size_t j = 0U;
  size_t i = xdim;
  size_t iminxdim = 0;

  for (auto lut_idx = 0U; lut_idx < lookUpTableSize; ++lut_idx)
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
rtk::HndImageIO::CanWriteFile(const char * itkNotUsed(FileNameToWrite))
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void
rtk::HndImageIO::Write(const void * itkNotUsed(buffer))
{
  // TODO?
}
