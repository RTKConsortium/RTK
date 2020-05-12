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

#include "rtkHncImageIO.h"
#include <itkMetaDataObject.h>

//--------------------------------------------------------------------
// Read Image Information
void
rtk::HncImageIO::ReadImageInformation()
{
  Hnc_header hnc;
  FILE *     fp;

  fp = fopen(m_FileName.c_str(), "rb");
  if (fp == nullptr)
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);

  size_t nelements = 0;
  nelements += fread((void *)hnc.sFileType, sizeof(char), 32, fp);
  nelements += fread((void *)&hnc.FileLength, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnc.sChecksumSpec, sizeof(char), 4, fp);
  nelements += fread((void *)&hnc.nCheckSum, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnc.sCreationDate, sizeof(char), 8, fp);
  nelements += fread((void *)hnc.sCreationTime, sizeof(char), 8, fp);
  nelements += fread((void *)hnc.sPatientID, sizeof(char), 16, fp);
  nelements += fread((void *)&hnc.nPatientSer, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnc.sSeriesID, sizeof(char), 16, fp);
  nelements += fread((void *)&hnc.nSeriesSer, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnc.sSliceID, sizeof(char), 16, fp);
  nelements += fread((void *)&hnc.nSliceSer, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnc.SizeX, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnc.SizeY, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnc.dSliceZPos, sizeof(double), 1, fp);
  nelements += fread((void *)hnc.sModality, sizeof(char), 16, fp);
  nelements += fread((void *)&hnc.nWindow, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnc.nLevel, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)&hnc.nPixelOffset, sizeof(itk::uint32_t), 1, fp);
  nelements += fread((void *)hnc.sImageType, sizeof(char), 4, fp);
  nelements += fread((void *)&hnc.dGantryRtn, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dSAD, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dSFD, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCollX1, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCollX2, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCollY1, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCollY2, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCollRtn, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dFieldX, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dFieldY, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dBladeX1, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dBladeX2, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dBladeY1, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dBladeY2, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dIDUPosLng, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dIDUPosLat, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dIDUPosVrt, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dIDUPosRtn, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dPatientSupportAngle, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dTableTopEccentricAngle, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCouchVrt, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCouchLng, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCouchLat, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dIDUResolutionX, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dIDUResolutionY, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dImageResolutionX, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dImageResolutionY, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dEnergy, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dDoseRate, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dXRayKV, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dXRayMA, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dMetersetExposure, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dAcqAdjustment, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCTProjectionAngle, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dCTNormChamber, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dGatingTimeTag, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dGating4DInfoX, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dGating4DInfoY, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dGating4DInfoZ, sizeof(double), 1, fp);
  nelements += fread((void *)&hnc.dGating4DInfoTime, sizeof(double), 1, fp);

  if (nelements != /*char*/ 120 + /*itk::uint32_t*/ 10 + /*double*/ 41)
    itkGenericExceptionMacro(<< "Could not read header data in " << m_FileName);

  if (fclose(fp) != 0)
    itkGenericExceptionMacro(<< "Could not close file: " << m_FileName);

  /* Convert hnc to ITK image information */
  SetNumberOfDimensions(2);
  SetDimensions(0, hnc.SizeX);
  SetDimensions(1, hnc.SizeY);
  SetSpacing(0, hnc.dIDUResolutionX);
  SetSpacing(1, hnc.dIDUResolutionY);
  SetOrigin(0, -0.5 * (hnc.SizeX - 1) * hnc.dIDUResolutionX); // SR: assumed centered
  SetOrigin(1, -0.5 * (hnc.SizeY - 1) * hnc.dIDUResolutionY); // SR: assumed centered
  SetComponentType(itk::ImageIOBase::IOComponentEnum::USHORT);

  /* Store important meta information in the meta data dictionary */
  itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTProjectionAngle", hnc.dCTProjectionAngle);
  itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTNormChamber", hnc.dCTNormChamber);
}

//--------------------------------------------------------------------
bool
rtk::HncImageIO::CanReadFile(const char * FileNameToRead)
{
  std::string                  filename(FileNameToRead);
  const std::string::size_type it = filename.find_last_of(".");
  std::string                  fileExt(filename, it + 1, filename.length());

  if (fileExt != std::string("hnc"))
    return false;
  return true;
}

//--------------------------------------------------------------------
// Read Image Content
void
rtk::HncImageIO::Read(void * buffer)
{
  FILE * fp;

  fp = fopen(m_FileName.c_str(), "rb");
  if (fp == nullptr)
  {
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);
    return;
  }

  if (fseek(fp, 512, SEEK_SET) != 0)
    itkGenericExceptionMacro(<< "Could not seek 512 bytes in " << m_FileName);

  size_t nelements = GetDimensions(0) * GetDimensions(1);
  if (fread(buffer, sizeof(unsigned short int), nelements, fp) != nelements)
    itkGenericExceptionMacro(<< "Could not read " << nelements << " bytes in " << m_FileName);

  fclose(fp);
  return;
}

//--------------------------------------------------------------------
bool
rtk::HncImageIO::CanWriteFile(const char * itkNotUsed(FileNameToWrite))
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void
rtk::HncImageIO::Write(const void * itkNotUsed(buffer))
{
  // TODO?
}
