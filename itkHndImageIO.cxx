// std include
#include <stdio.h>

#include "itkHndImageIO.h"
#include "itkMetaDataObject.h"

//--------------------------------------------------------------------
// Read Image Information
void itk::HndImageIO::ReadImageInformation()
{
  Hnd_header hnd;
  FILE *fp;

  fp = fopen (m_FileName.c_str(), "rb");
  if (fp == NULL)
      itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);

  fread ((void *) hnd.sFileType, sizeof(char), 32, fp);
  fread ((void *) &hnd.FileLength, sizeof(uint32_t), 1, fp);
  fread ((void *) hnd.sChecksumSpec, sizeof(char), 4, fp);
  fread ((void *) &hnd.nCheckSum, sizeof(uint32_t), 1, fp);
  fread ((void *) hnd.sCreationDate, sizeof(char), 8, fp);
  fread ((void *) hnd.sCreationTime, sizeof(char), 8, fp);
  fread ((void *) hnd.sPatientID, sizeof(char), 16, fp);
  fread ((void *) &hnd.nPatientSer, sizeof(uint32_t), 1, fp);
  fread ((void *) hnd.sSeriesID, sizeof(char), 16, fp);
  fread ((void *) &hnd.nSeriesSer, sizeof(uint32_t), 1, fp);
  fread ((void *) hnd.sSliceID, sizeof(char), 16, fp);
  fread ((void *) &hnd.nSliceSer, sizeof(uint32_t), 1, fp);
  fread ((void *) &hnd.SizeX, sizeof(uint32_t), 1, fp);
  fread ((void *) &hnd.SizeY, sizeof(uint32_t), 1, fp);
  fread ((void *) &hnd.dSliceZPos, sizeof(double), 1, fp);
  fread ((void *) hnd.sModality, sizeof(char), 16, fp);
  fread ((void *) &hnd.nWindow, sizeof(uint32_t), 1, fp);
  fread ((void *) &hnd.nLevel, sizeof(uint32_t), 1, fp);
  fread ((void *) &hnd.nPixelOffset, sizeof(uint32_t), 1, fp);
  fread ((void *) hnd.sImageType, sizeof(char), 4, fp);
  fread ((void *) &hnd.dGantryRtn, sizeof(double), 1, fp);
  fread ((void *) &hnd.dSAD, sizeof(double), 1, fp);
  fread ((void *) &hnd.dSFD, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCollX1, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCollX2, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCollY1, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCollY2, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCollRtn, sizeof(double), 1, fp);
  fread ((void *) &hnd.dFieldX, sizeof(double), 1, fp);
  fread ((void *) &hnd.dFieldY, sizeof(double), 1, fp);
  fread ((void *) &hnd.dBladeX1, sizeof(double), 1, fp);
  fread ((void *) &hnd.dBladeX2, sizeof(double), 1, fp);
  fread ((void *) &hnd.dBladeY1, sizeof(double), 1, fp);
  fread ((void *) &hnd.dBladeY2, sizeof(double), 1, fp);
  fread ((void *) &hnd.dIDUPosLng, sizeof(double), 1, fp);
  fread ((void *) &hnd.dIDUPosLat, sizeof(double), 1, fp);
  fread ((void *) &hnd.dIDUPosVrt, sizeof(double), 1, fp);
  fread ((void *) &hnd.dIDUPosRtn, sizeof(double), 1, fp);
  fread ((void *) &hnd.dPatientSupportAngle, sizeof(double), 1, fp);
  fread ((void *) &hnd.dTableTopEccentricAngle, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCouchVrt, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCouchLng, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCouchLat, sizeof(double), 1, fp);
  fread ((void *) &hnd.dIDUResolutionX, sizeof(double), 1, fp);
  fread ((void *) &hnd.dIDUResolutionY, sizeof(double), 1, fp);
  fread ((void *) &hnd.dImageResolutionX, sizeof(double), 1, fp);
  fread ((void *) &hnd.dImageResolutionY, sizeof(double), 1, fp);
  fread ((void *) &hnd.dEnergy, sizeof(double), 1, fp);
  fread ((void *) &hnd.dDoseRate, sizeof(double), 1, fp);
  fread ((void *) &hnd.dXRayKV, sizeof(double), 1, fp);
  fread ((void *) &hnd.dXRayMA, sizeof(double), 1, fp);
  fread ((void *) &hnd.dMetersetExposure, sizeof(double), 1, fp);
  fread ((void *) &hnd.dAcqAdjustment, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCTProjectionAngle, sizeof(double), 1, fp);
  fread ((void *) &hnd.dCTNormChamber, sizeof(double), 1, fp);
  fread ((void *) &hnd.dGatingTimeTag, sizeof(double), 1, fp);
  fread ((void *) &hnd.dGating4DInfoX, sizeof(double), 1, fp);
  fread ((void *) &hnd.dGating4DInfoY, sizeof(double), 1, fp);
  fread ((void *) &hnd.dGating4DInfoZ, sizeof(double), 1, fp);
  fread ((void *) &hnd.dGating4DInfoTime, sizeof(double), 1, fp);
  fclose (fp);

  /* Convert hnd to ITK image information */
  SetNumberOfDimensions(2);
  SetDimensions(0, hnd.SizeX);
  SetDimensions(1, hnd.SizeY);
  SetSpacing(0, hnd.dIDUResolutionX);
  SetSpacing(1, hnd.dIDUResolutionY);
  SetOrigin(0, -0.5*(hnd.SizeX-1)*hnd.dIDUResolutionX); //SR: assumed centered
  SetOrigin(1, -0.5*(hnd.SizeY-1)*hnd.dIDUResolutionY); //SR: assumed centered
  SetComponentType(itk::ImageIOBase::UINT);

  /* Store important meta information in the meta data dictionary */
  itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTProjectionAngle", hnd.dCTProjectionAngle);
  //itk::ExposeMetaData<double>( this->GetMetaDataDictionary(), &(hnd.dCTProjectionAngle), "dCTProjectionAngle");
}

//--------------------------------------------------------------------
bool itk::HndImageIO::CanReadFile(const char* FileNameToRead)
{
  std::string filename(FileNameToRead);
  const std::string::size_type it = filename.find_last_of( "." );
  std::string fileExt( filename, it+1, filename.length() );
  if (fileExt != std::string("hnd"))
    return false;
  return true;
}

//--------------------------------------------------------------------
// Read Image Content
void itk::HndImageIO::Read(void * buffer)
{
    FILE *fp;

    uint32_t* buf = (uint32_t*)buffer;
    unsigned char *pt_lut;
    uint32_t a;
    float b;
    unsigned char v;
    int lut_idx, lut_off;
    size_t num_read;
    char dc;
    short ds;
    long dl, diff=0;
    uint32_t i;

    fp = fopen (m_FileName.c_str(), "rb");
    if (fp == NULL)
        itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);

    pt_lut = (unsigned char*) malloc (sizeof (unsigned char) * GetDimensions(0) * GetDimensions(1));

    /* Read LUT */
    fseek (fp, 1024, SEEK_SET);
    fread (pt_lut, sizeof(unsigned char), (GetDimensions(1)-1)*GetDimensions(0) / 4, fp);

    /* Read first row */
    for (i = 0; i < GetDimensions(0); i++) {
        fread (&a, sizeof(uint32_t), 1, fp);
        buf[i] = a;
        b = a;
    }

    /* Read first pixel of second row */
    fread (&a, sizeof(uint32_t), 1, fp);
    buf[i++] = a;
    b = a;

    /* Decompress the rest */
    lut_idx = 0;
    lut_off = 0;
    while (i < GetDimensions(0) * GetDimensions(1)) {
        uint32_t r11, r12, r21;

        r11 = buf[i-GetDimensions(0)-1];
        r12 = buf[i-GetDimensions(0)];
        r21 = buf[i-1];
        v = pt_lut[lut_idx];
        switch (lut_off) {
        case 0:
            v = v & 0x03;
            lut_off ++;
            break;
        case 1:
            v = (v & 0x0C) >> 2;
            lut_off ++;
            break;
        case 2:
            v = (v & 0x30) >> 4;
            lut_off ++;
            break;
        case 3:
            v = (v & 0xC0) >> 6;
            lut_off = 0;
            lut_idx ++;
            break;
        }
        switch (v) {
        case 0:
            num_read = fread (&dc, sizeof(unsigned char), 1, fp);
            if (num_read != 1) goto read_error;
            diff = dc;
            break;
        case 1:
            num_read = fread (&ds, sizeof(unsigned short), 1, fp);
            if (num_read != 1) goto read_error;
            diff = ds;
            break;
        case 2:
            num_read = fread (&dl, sizeof(uint32_t), 1, fp);
            if (num_read != 1) goto read_error;
            diff = dl;
            break;
        }

        buf[i] = r21 + r12 + diff - r11;
        b = buf[i];
        i++;
    }

    /* Clean up */
    free (pt_lut);
    fclose (fp);
    return;

 read_error:

    itkGenericExceptionMacro(<< "Error reading hnd file");
    free (pt_lut);
    fclose (fp);
    return;
}

//--------------------------------------------------------------------
bool itk::HndImageIO::CanWriteFile(const char* FileNameToWrite)
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void itk::HndImageIO::Write(const void* buffer)
{
  //TODO?
}
