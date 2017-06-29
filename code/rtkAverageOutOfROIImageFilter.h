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
#ifndef rtkAverageOutOfROIImageFilter_h
#define rtkAverageOutOfROIImageFilter_h

#include "itkInPlaceImageFilter.h"

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  #include <itkImageRegionSplitterDirection.h>
#endif

#include "rtkMacro.h"

namespace rtk
{
  /** \class AverageOutOfROIImageFilter
   * \brief Averages along the last dimension if the pixel is outside ROI
   *
   * This filter takes in input a n-D image and an (n-1)D binary image
   * representing a region of interest (1 inside the ROI, 0 outside).
   * The filter walks through the ROI image, and :
   * - if it contains 0, pixels in the n-D image a replaced with their
   * average along the last dimension
   * - if it contains 1, nothing happens
   *
   * This filter is used in rtk4DROOSTERConeBeamReconstructionFilter in
   * order to average along time between phases, everywhere except where
   * movement is expected to occur.
   *
   * \test rtkfourdroostertest.cxx
   *
   * \author Cyril Mory
   *
   */
template< class TInputImage,
          class TROI = itk::Image< typename TInputImage::PixelType, TInputImage::ImageDimension -1 > >

class AverageOutOfROIImageFilter : public itk::InPlaceImageFilter<TInputImage, TInputImage>
{
public:
    /** Standard class typedefs. */
    typedef AverageOutOfROIImageFilter                        Self;
    typedef itk::ImageToImageFilter<TInputImage, TInputImage> Superclass;
    typedef itk::SmartPointer< Self >                         Pointer;
    typedef itk::Image<typename TInputImage::PixelType, TInputImage::ImageDimension - 1>       LowerDimImage;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(AverageOutOfROIImageFilter, itk::InPlaceImageFilter)

    /** The image containing the weights applied to the temporal components */
    void SetROI(const TROI* Map);

protected:
    AverageOutOfROIImageFilter();
    ~AverageOutOfROIImageFilter() {}

    typename TROI::Pointer GetROI();

    void GenerateOutputInformation() ITK_OVERRIDE;
    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Does the real work. */
    void ThreadedGenerateData(const typename TInputImage::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;
    void BeforeThreadedGenerateData() ITK_OVERRIDE;

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
    /** Splits the OutputRequestedRegion along the first direction, not the last */
    const itk::ImageRegionSplitterBase* GetImageRegionSplitter(void) const ITK_OVERRIDE;
    itk::ImageRegionSplitterDirection::Pointer  m_Splitter;
#endif

private:
    AverageOutOfROIImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkAverageOutOfROIImageFilter.hxx"
#endif

#endif
