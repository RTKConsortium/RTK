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

#ifndef rtkHndImageIO_h
#define rtkHndImageIO_h

// itk include
#include <itkImageIOBase.h>

#if defined (_MSC_VER) && (_MSC_VER < 1600)
//SR: taken from
//#include "msinttypes/stdint.h"
#else
#include <stdint.h>
#endif

#include "rtkMacro.h"

namespace rtk {

/** \class HndImageIO
 * \brief Class for reading Hnd Image file format
 *
 * Reads Hnd files (file format used by Varian for Obi raw data).
 *
 * \author Simon Rit
 *
 * \ingroup IOFilters
 */
class HndImageIO : public itk::ImageIOBase
{
public:
/** Standard class typedefs. */
  typedef HndImageIO              Self;
  typedef itk::ImageIOBase        Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef signed short int        PixelType;

  typedef struct hnd_header {
    char sFileType[32];
    unsigned int FileLength;
    char sChecksumSpec[4];
    unsigned int nCheckSum;
    char sCreationDate[8];
    char sCreationTime[8];
    char sPatientID[16];
    unsigned int nPatientSer;
    char sSeriesID[16];
    unsigned int nSeriesSer;
    char sSliceID[16];
    unsigned int nSliceSer;
    unsigned int SizeX;
    unsigned int SizeY;
    double dSliceZPos;
    char sModality[16];
    unsigned int nWindow;
    unsigned int nLevel;
    unsigned int nPixelOffset;
    char sImageType[4];
    double dGantryRtn;
    double dSAD;
    double dSFD;
    double dCollX1;
    double dCollX2;
    double dCollY1;
    double dCollY2;
    double dCollRtn;
    double dFieldX;
    double dFieldY;
    double dBladeX1;
    double dBladeX2;
    double dBladeY1;
    double dBladeY2;
    double dIDUPosLng;
    double dIDUPosLat;
    double dIDUPosVrt;
    double dIDUPosRtn;
    double dPatientSupportAngle;
    double dTableTopEccentricAngle;
    double dCouchVrt;
    double dCouchLng;
    double dCouchLat;
    double dIDUResolutionX;
    double dIDUResolutionY;
    double dImageResolutionX;
    double dImageResolutionY;
    double dEnergy;
    double dDoseRate;
    double dXRayKV;
    double dXRayMA;
    double dMetersetExposure;
    double dAcqAdjustment;
    double dCTProjectionAngle;
    double dCTNormChamber;
    double dGatingTimeTag;
    double dGating4DInfoX;
    double dGating4DInfoY;
    double dGating4DInfoZ;
    double dGating4DInfoTime;
    } Hnd_header;

  HndImageIO() : Superclass() {}

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(HndImageIO, itk::ImageIOBase);

  /*-------- This part of the interface deals with reading data. ------ */
  void ReadImageInformation() ITK_OVERRIDE;

  bool CanReadFile( const char* FileNameToRead ) ITK_OVERRIDE;

  void Read(void * buffer) ITK_OVERRIDE;

  /*-------- This part of the interfaces deals with writing data. ----- */
  virtual void WriteImageInformation(bool /*keepOfStream*/) { }

  void WriteImageInformation() ITK_OVERRIDE { WriteImageInformation(false); }

  bool CanWriteFile(const char* filename) ITK_OVERRIDE;

  void Write(const void* buffer) ITK_OVERRIDE;

}; // end class HndImageIO

} // end namespace

#endif /* end #define rtkHndImageIO_h */
