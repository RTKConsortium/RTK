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

#ifndef rtkMedianImageFilter_h
#define rtkMedianImageFilter_h

#include <itkImageToImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class MedianImageFilter
 * \brief Performes a Median filtering on a 2D image of pixel type unsigned short (16bits).
 *
 * Performs a median filtering with different possible windows ( 3x3 or 3x2 )
 * A median filter consists of replacing each entry pixel with the median of
 * neighboring pixels. The number of neighboring pixels depends on the windows
 * size.
 *
 * \test rtkmediantest.cxx
 *
 * \author S. Brousmiche (UCL-iMagX) and Marc Vila
 *
 * \ingroup ImageToImageFilter
 */

//FIXME: not templated yet
//template <class TInputImage, class TOutputImage>
typedef itk::Image<unsigned short, 2>  TImage;
class ITK_EXPORT MedianImageFilter:
  public itk::ImageToImageFilter< TImage, TImage >
{
public:
  /** Standard class typedefs. */
  typedef MedianImageFilter                       Self;
  typedef itk::ImageToImageFilter<TImage,TImage>  Superclass;
  typedef itk::SmartPointer<Self>                 Pointer;
  typedef itk::SmartPointer<const Self>           ConstPointer;

  typedef TImage::PixelType                 *OutputImagePointer;
  typedef TImage::PixelType                 *InputImagePointer;
  typedef TImage::RegionType                OutputImageRegionType;

  typedef itk::Vector<unsigned int, TImage::ImageDimension> VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MedianImageFilter, ImageToImageFilter);

  /** Get / Set the Median window that are going to be used during the operation */
  itkGetMacro(MedianWindow, VectorType);
  itkSetMacro(MedianWindow, VectorType);

protected:
  MedianImageFilter();
  ~MedianImageFilter() {}

  //virtual void GenerateOutputInformation();
  //virtual void GenerateInputRequestedRegion();

  /** Performs median filtering on input image.
   * A call to this function will assume modification of the function.*/
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

private:
  MedianImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  VectorType m_MedianWindow;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkMedianImageFilter.cxx"
#endif

#endif
