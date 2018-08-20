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
  typedef itk::int32_t            Int4; // int of 4 bytes as in xim docs

  typedef struct xim_header {
    //Actual Header:
    char sFileType[32];
    Int4 FileVersion;
    Int4 SizeX;
    Int4 SizeY;
    Int4 dBitsPerPixel;
    Int4 dBytesPerPixel;
    Int4 dCompressionIndicator;
    Int4 lookUpTableSize;
    Int4 compressedPixelBufferSize;
    Int4 unCompressedPixelBufferSize;
    //Header after pixel-data:
    Int4 binsInHistogram;
    Int4 * histogramData;
    Int4 numberOfProperties;
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
  void ReadImageInformation() ITK_OVERRIDE;

  bool CanReadFile( const char* FileNameToRead ) ITK_OVERRIDE;

  void Read(void * buffer) ITK_OVERRIDE;

  /*-------- This part of the interfaces deals with writing data. ----- */
  virtual void WriteImageInformation(bool /*keepOfStream*/) { }

  void WriteImageInformation()  ITK_OVERRIDE { WriteImageInformation(false); }

  bool CanWriteFile(const char* filename) ITK_OVERRIDE;

  void Write(const void* buffer) ITK_OVERRIDE;

private:
  template<typename T> size_t SetPropertyValue(char *property_name, Int4 value_length, FILE *fp, Xim_header *xim);
  template<typename T> static T get_diff(char vsub, FILE* &fp);

  int          m_ImageDataStart;
  Int4 m_BytesPerPixel;

}; // end class XimImageIO

} // end namespace

#endif /* end #define __rtkXimImageIO_h */
