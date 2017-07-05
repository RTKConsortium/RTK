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
#ifndef rtkVectorImageToImageFilter_h
#define rtkVectorImageToImageFilter_h

#include "rtkMacro.h"

#include <itkImageToImageFilter.h>
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  #include <itkImageRegionSplitterDirection.h>
#endif

namespace rtk
{
  /** \class VectorImageToImageFilter
   * \brief Re-writes a vector image as an image
   *
   * Depending on the dimensions of the input and output images, the filter
   * can have two different behaviors:
   *  - if the dimensions match, the channels of the input image are
   * concatenated in the last dimension. With an input image of size (X,Y)
   * containing N-components vectors, the output image will be of size (X, Y*N)
   *  - if the output image dimension equals to the input dimension plus one,
   * the additional dimension of the output image will contain the channels of the input.
   * With an input image of size (X,Y) containing N-components vectors,
   * the output image will be of size (X, Y, N).
   *
   * \author Cyril Mory
   */

template< typename InputImageType, typename OutputImageType>
class VectorImageToImageFilter : public itk::ImageToImageFilter< InputImageType, OutputImageType >
{
public:
    /** Standard class typedefs. */
    typedef VectorImageToImageFilter                                    Self;
    typedef itk::ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
    typedef itk::SmartPointer< Self >                                   Pointer;

    typedef typename OutputImageType::RegionType                        OutputImageRegionType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(VectorImageToImageFilter, itk::ImageToImageFilter)

protected:
    VectorImageToImageFilter();
    ~VectorImageToImageFilter() {}

    void GenerateOutputInformation() ITK_OVERRIDE;
    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Does the real work. */
    void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;
    void BeforeThreadedGenerateData() ITK_OVERRIDE;

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
    /** Splits the OutputRequestedRegion along the first direction, not the last */
    const itk::ImageRegionSplitterBase* GetImageRegionSplitter(void) const ITK_OVERRIDE;
    itk::ImageRegionSplitterDirection::Pointer  m_Splitter;
#endif

private:
    VectorImageToImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkVectorImageToImageFilter.hxx"
#endif

#endif
