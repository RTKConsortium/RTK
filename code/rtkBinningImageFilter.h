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

#ifndef __rtkBinningImageFilter_h
#define __rtkBinningImageFilter_h

#include <itkImageToImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class BinningImageFilter
 * \brief Performes a binning on a 2D image of pixel type unsigned short (16bits).
 *
 * Groups of pixels are combined to form larger single pixels.
 * For example, when there is a 2x2 binning, before you had 4 pixels
 * but after binning you receive information for a single pixel.
 * Each new pixel contains all the information from the previous
 * original pixels.  This makes the new pixel 4 times brighter,
 * but the image also has 4 times less pixels.
 *
 * \test rtkbinningtest.cxx
 *
 * \author S. Brousmiche (UCL-iMagX) and Marc Vila
 *
 * \ingroup ImageToImageFilter
 */

class ITK_EXPORT BinningImageFilter:
  public itk::ImageToImageFilter< itk::Image<unsigned short, 2>, itk::Image<unsigned short, 2> >
{
public:
  /** Standard class typedefs. */
  typedef BinningImageFilter                                Self;
  typedef itk::Image<unsigned short, 2>                     ImageType;
  typedef itk::ImageToImageFilter<ImageType, ImageType>     Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  typedef ImageType::PixelType                 *OutputImagePointer;
  typedef ImageType::PixelType                 *InputImagePointer;
  typedef ImageType::RegionType                OutputImageRegionType;

  typedef itk::Vector<unsigned int, ImageType::ImageDimension> VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BinningImageFilter, ImageToImageFilter);

  /** Get / Set the binning factors that are going to be used during the operation */
  itkGetMacro(BinningFactors, VectorType);
  itkSetMacro(BinningFactors, VectorType);

protected:
  BinningImageFilter();
  virtual ~BinningImageFilter() {};

  virtual void GenerateOutputInformation();
  virtual void GenerateInputRequestedRegion();

  /** Performs a binning on input image.
   * A call to this function will assume modification of the function.*/
  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

private:
  BinningImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  VectorType m_BinningFactors;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkBinningImageFilter.cxx"
#endif

#endif
