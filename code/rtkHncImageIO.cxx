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
#include <stdio.h>

#include "rtkHncImageIO.h"
#include "itkMetaDataObject.h"


static std::string
GetExtension(const std::string & filename)
{
  std::string fileExt( itksys::SystemTools::GetFilenameLastExtension(filename) );

  //If the last extension is .bz2, then need to pull off 2 extensions.
  //.bz2 is the only valid compression extension.
  if ( fileExt == std::string(".bz2") )
    {
    fileExt =
      itksys::SystemTools::GetFilenameLastExtension( itksys::SystemTools::GetFilenameWithoutLastExtension(filename) );
    fileExt += ".bz2";
    }
  //Check that a valid extension was found.
  if ( fileExt != ".hnc.bz2" && fileExt != ".hnc" )
    {
    return ( "" );
    }
  return ( fileExt );
}

//--------------------------------------------------------------------
// Read Image Information
void rtk::HncImageIO::ReadImageInformation()
{
  Hnc_header hnc;
  FILE *     fp;

  fp = fopen (m_FileName.c_str(), "rb");
  if (fp == NULL)
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);

  const std::string fileExt = GetExtension(m_FileName);
  
  if( fileExt == ".hnc" )
    {
	  fread ( (void *) hnc.sFileType, sizeof(char), 32, fp);
	  fread ( (void *) &hnc.FileLength, sizeof(uint32_t), 1, fp);
	  fread ( (void *) hnc.sChecksumSpec, sizeof(char), 4, fp);
	  fread ( (void *) &hnc.nCheckSum, sizeof(uint32_t), 1, fp);
	  fread ( (void *) hnc.sCreationDate, sizeof(char), 8, fp);
	  fread ( (void *) hnc.sCreationTime, sizeof(char), 8, fp);
	  fread ( (void *) hnc.sPatientID, sizeof(char), 16, fp);
	  fread ( (void *) &hnc.nPatientSer, sizeof(uint32_t), 1, fp);
	  fread ( (void *) hnc.sSeriesID, sizeof(char), 16, fp);
	  fread ( (void *) &hnc.nSeriesSer, sizeof(uint32_t), 1, fp);
	  fread ( (void *) hnc.sSliceID, sizeof(char), 16, fp);
	  fread ( (void *) &hnc.nSliceSer, sizeof(uint32_t), 1, fp);
	  fread ( (void *) &hnc.SizeX, sizeof(uint32_t), 1, fp);
	  fread ( (void *) &hnc.SizeY, sizeof(uint32_t), 1, fp);
	  fread ( (void *) &hnc.dSliceZPos, sizeof(double), 1, fp);
	  fread ( (void *) hnc.sModality, sizeof(char), 16, fp);
	  fread ( (void *) &hnc.nWindow, sizeof(uint32_t), 1, fp);
	  fread ( (void *) &hnc.nLevel, sizeof(uint32_t), 1, fp);
	  fread ( (void *) &hnc.nPixelOffset, sizeof(uint32_t), 1, fp);
	  fread ( (void *) hnc.sImageType, sizeof(char), 4, fp);
	  fread ( (void *) &hnc.dGantryRtn, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dSAD, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dSFD, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCollX1, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCollX2, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCollY1, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCollY2, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCollRtn, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dFieldX, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dFieldY, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dBladeX1, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dBladeX2, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dBladeY1, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dBladeY2, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dIDUPosLng, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dIDUPosLat, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dIDUPosVrt, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dIDUPosRtn, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dPatientSupportAngle, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dTableTopEccentricAngle, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCouchVrt, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCouchLng, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCouchLat, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dIDUResolutionX, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dIDUResolutionY, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dImageResolutionX, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dImageResolutionY, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dEnergy, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dDoseRate, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dXRayKV, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dXRayMA, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dMetersetExposure, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dAcqAdjustment, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCTProjectionAngle, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dCTNormChamber, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dGatingTimeTag, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dGating4DInfoX, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dGating4DInfoY, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dGating4DInfoZ, sizeof(double), 1, fp);
	  fread ( (void *) &hnc.dGating4DInfoTime, sizeof(double), 1, fp);
	  fclose (fp);
    }
  else if( fileExt == ".hnc.bz2" )
    {
	  BZFILE* bp;
	  int nBuf;
	  int bzerror;
	  
	  bp = BZ2_bzReadOpen ( &bzerror, fp, 0, 0, NULL, 0 );
	  if ( bzerror != BZ_OK ) {
        BZ2_bzReadClose ( &bzerror, bp );
	    itkGenericExceptionMacro(<< "Could not read file:" << m_FileName);
      }

	  //nBuf = BZ2_bzRead ( &bzerror, bp, buf, 512 );
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) hnc.sFileType, sizeof(char)* 32);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.FileLength, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) hnc.sChecksumSpec, sizeof(char)* 4);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.nCheckSum, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) hnc.sCreationDate, sizeof(char)* 8);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) hnc.sCreationTime, sizeof(char)* 8);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) hnc.sPatientID, sizeof(char)* 16);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.nPatientSer, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) hnc.sSeriesID, sizeof(char)* 16);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.nSeriesSer, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) hnc.sSliceID, sizeof(char)* 16);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.nSliceSer, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.SizeX, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.SizeY, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dSliceZPos, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) hnc.sModality, sizeof(char)* 16);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.nWindow, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.nLevel, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.nPixelOffset, sizeof(uint32_t)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) hnc.sImageType, sizeof(char)* 4);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dGantryRtn, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dSAD, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dSFD, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCollX1, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCollX2, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCollY1, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCollY2, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCollRtn, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dFieldX, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dFieldY, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dBladeX1, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dBladeX2, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dBladeY1, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dBladeY2, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dIDUPosLng, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dIDUPosLat, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dIDUPosVrt, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dIDUPosRtn, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dPatientSupportAngle, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dTableTopEccentricAngle, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCouchVrt, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCouchLng, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCouchLat, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dIDUResolutionX, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dIDUResolutionY, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dImageResolutionX, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dImageResolutionY, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dEnergy, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dDoseRate, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dXRayKV, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dXRayMA, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dMetersetExposure, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dAcqAdjustment, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCTProjectionAngle, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dCTNormChamber, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dGatingTimeTag, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dGating4DInfoX, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dGating4DInfoY, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dGating4DInfoZ, sizeof(double)* 1);
	  nBuf = BZ2_bzRead ( &bzerror, bp,  (void *) &hnc.dGating4DInfoTime, sizeof(double)* 1);
	  
	  BZ2_bzReadClose ( &bzerror, bp );
	  fclose (fp);
	}
  else
    itkGenericExceptionMacro(<< "Could not read file: " << m_FileName);
	
  /* Convert hnc to ITK image information */
  SetNumberOfDimensions(2);
  SetDimensions(0, hnc.SizeX);
  SetDimensions(1, hnc.SizeY);
  SetSpacing(0, hnc.dIDUResolutionX);
  SetSpacing(1, hnc.dIDUResolutionY);
  SetOrigin(0, -0.5*(hnc.SizeX-1)*hnc.dIDUResolutionX); //SR: assumed centered
  SetOrigin(1, -0.5*(hnc.SizeY-1)*hnc.dIDUResolutionY); //SR: assumed centered
  SetComponentType(itk::ImageIOBase::USHORT);

  /* Store important meta information in the meta data dictionary */
  itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTProjectionAngle", hnc.dCTProjectionAngle);
  itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTNormChamber", hnc.dCTNormChamber);

}

//--------------------------------------------------------------------
bool rtk::HncImageIO::CanReadFile(const char* FileNameToRead)
{
  std::string filename(FileNameToRead);
  std::string fileExt( itksys::SystemTools::GetFilenameLastExtension(filename) );

  if ( fileExt == std::string(".bz2") )
    {
    fileExt =
      itksys::SystemTools::GetFilenameLastExtension( itksys::SystemTools::GetFilenameWithoutLastExtension(filename) );
    fileExt += ".bz2";
    }
  //Check that a valid extension was found.
  if ( fileExt != ".hnc.bz2" && fileExt != ".hnc" )
    {
    return false;
    }

  return true;
}

//--------------------------------------------------------------------
// Read Image Content
void rtk::HncImageIO::Read(void * buffer)
{
  FILE *     fp;

  fp = fopen (m_FileName.c_str(), "rb");
  if (fp == NULL)
    {
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);
    return;
    }
	
  const std::string fileExt = GetExtension(m_FileName);
  
  if( fileExt == ".hnc" )
    {
	  fseek(fp, 512, SEEK_SET);
	  fread(buffer, sizeof(unsigned short int), GetDimensions(0)*GetDimensions(1), fp);
    }
  else if( fileExt == ".hnc.bz2" )
    {
	  BZFILE* bp;
	  int nBuf;
	  int bzerror;
	  
	  bp = BZ2_bzReadOpen ( &bzerror, fp, 0, 0, NULL, 0 );
	  if ( bzerror != BZ_OK ) {
        BZ2_bzReadClose ( &bzerror, bp );
	    itkGenericExceptionMacro(<< "Could not read file:" << m_FileName);
      }
	  char hdr[512];
	  nBuf = BZ2_bzRead ( &bzerror, bp, (void*) hdr, 512);
	  nBuf = BZ2_bzRead ( &bzerror, bp, buffer, sizeof(unsigned short int) * GetDimensions(0) * GetDimensions(1));
	  
	  BZ2_bzReadClose ( &bzerror, bp );
  	}

/*  file.seekg(512, std::ios::beg);
  if( !this->ReadBufferAsBinary( file, buffer, sizeof(unsigned short int) * GetDimensions(0) * GetDimensions(1)) )
    {
    itkExceptionMacro(<< "Could not read file: " << m_FileName);
    file.close();
    return;
    }
*/
  fclose (fp);
  return;

}

//--------------------------------------------------------------------
bool rtk::HncImageIO::CanWriteFile(const char* FileNameToWrite)
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void rtk::HncImageIO::Write(const void* buffer)
{
  //TODO?
}
