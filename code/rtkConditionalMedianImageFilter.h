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
#ifndef rtkConditionalMedianImageFilter_h
#define rtkConditionalMedianImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkConstNeighborhoodIterator.h>
#include <itkVectorImage.h>

#include "rtkWin32Header.h"
#include "rtkMacro.h"

namespace rtk
{
/** \class ConditionalMedianImageFilter
 * \brief Performs a median filtering on outlier pixels
 *
 * ConditionalMedianImageFilter computes the median of the pixel values
 * in a neighborhood around each pixel. If the input pixel value is close
 * to the computed median, it is kept unchanged and copied to the output.
 * Otherwise it is replaced by the computed median.
 * Note that if m_ThresholdMultiplier = 0, this filter behaves like a usual
 * median filter, and if m_Radius = [0, 0, ..., 0], the image passes through
 * unchanged.
 *
 * \test TODO
 *
 * \author Cyril Mory
 *
 */
template< typename TInputImage>
class ConditionalMedianImageFilter : public itk::InPlaceImageFilter<TInputImage>
{
public:
    /** Standard class typedefs. */
    typedef ConditionalMedianImageFilter                                        Self;
    typedef itk::InPlaceImageFilter<TInputImage>                                Superclass;
    typedef itk::SmartPointer< Self >                                           Pointer;
    typedef typename itk::ConstNeighborhoodIterator<TInputImage>::RadiusType    MedianRadiusType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ConditionalMedianImageFilter, itk::InPlaceImageFilter)

    /** Set/Get neighborhood radius */
    itkSetMacro(Radius, MedianRadiusType)
    itkGetMacro(Radius, MedianRadiusType)

    /** Set/Get neighborhood radius */
    itkSetMacro(ThresholdMultiplier, double)
    itkGetMacro(ThresholdMultiplier, double)

protected:
    ConditionalMedianImageFilter();
    ~ConditionalMedianImageFilter() {}

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Does the real work. */
    void ThreadedGenerateData(const typename TInputImage::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;

    MedianRadiusType  m_Radius;
    double            m_ThresholdMultiplier;

private:
    ConditionalMedianImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};

template <>
RTK_EXPORT
void
ConditionalMedianImageFilter<itk::VectorImage<float, 3> >
::ThreadedGenerateData(const itk::VectorImage<float, 3>::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId));

} //namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConditionalMedianImageFilter.hxx"
#endif

#endif // rtkConditionalMedianImageFilter_h
