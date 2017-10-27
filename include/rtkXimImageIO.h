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

#ifndef rtkXimImageIO_h
#define rtkXimImageIO_h

#include "rtkMacro.h"

// itk include
#include <itkImageIOBase.h>

#if defined (_MSC_VER) && (_MSC_VER < 1600)
//SR: taken from
//#include "msinttypes/stdint.h"
#else
#include <stdint.h>
#endif

namespace rtk {

/** \class XimImageIO
 * \brief Class for reading Xim Image file format
 *
 * Reads Xim files (file format used by Varian for Obi raw data).
 *
 * \author Andreas Gravgaard Andersen
 *
 * \ingroup IOFilters
 */
class XimImageIO : public itk::ImageIOBase
{
public:
/** Standard class typedefs. */
  typedef XimImageIO              Self;
  typedef itk::ImageIOBase        Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef signed short int        PixelType;

  typedef struct xim_header {
    //Actual Header:
    char sFileType[32];
    itk::int32_t FileVersion;
    itk::int32_t SizeX;
    itk::int32_t SizeY;
    itk::int32_t dBitsPerPixel;
    itk::int32_t dBytesPerPixel;
    itk::int32_t dCompressionIndicator;
    itk::int32_t lookUpTableSize;
    itk::int32_t compressedPixelBufferSize;
    itk::int32_t unCompressedPixelBufferSize;
    //Header after pixel-data:
    itk::int32_t binsInHistogram;
    itk::int32_t * histogramData;
    itk::int32_t numberOfProperties;
    unsigned int nPixelOffset;
    double dCollX1;
    double dCollX2;
    double dCollY1;
    double dCollY2;
    double dCollRtn;
    double dCouchVrt;
    double dCouchLng;
    double dCouchLat;
    double dIDUResolutionX; //MUST BE CALCULATED
    double dIDUResolutionY; //
    double dImageResolutionX;//
    double dImageResolutionY;//
    double dEnergy;
    double dDoseRate;
    double dXRayKV;
    double dXRayMA;
    double dCTProjectionAngle; //KVSourceRtn in file
    double dDetectorOffsetX; // KVDetectorLat
    double dDetectorOffsetY; // KVDetectorLng
    double dCTNormChamber;
    double dGatingTimeTag;
    double dGating4DInfoX;
    double dGating4DInfoY;
    double dGating4DInfoZ;
    //double dGating4DInfoTime;
    } Xim_header;

  XimImageIO() : Superclass() {}

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(XimImageIO, itk::ImageIOBase);

  /*-------- This part of the interface deals with reading data. ------ */
  virtual void ReadImageInformation() ITK_OVERRIDE;

  virtual bool CanReadFile( const char* FileNameToRead ) ITK_OVERRIDE;

  virtual void Read(void * buffer) ITK_OVERRIDE;

  /*-------- This part of the interfaces deals with writing data. ----- */
  virtual void WriteImageInformation(bool /*keepOfStream*/) { }

  virtual void WriteImageInformation()  ITK_OVERRIDE { WriteImageInformation(false); }

  virtual bool CanWriteFile(const char* filename) ITK_OVERRIDE;

  virtual void Write(const void* buffer) ITK_OVERRIDE;

private:
  template<typename T> size_t SetPropertyValue(char *property_name, itk::uint32_t value_length, FILE *fp, Xim_header *xim);

  int          m_ImageDataStart;
  itk::int32_t m_BytesPerPixel;

}; // end class XimImageIO

} // end namespace

#endif /* end #define __rtkXimImageIO_h */
