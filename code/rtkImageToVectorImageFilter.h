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
#ifndef rtkImageToVectorImageFilter_h
#define rtkImageToVectorImageFilter_h

#include "rtkMacro.h"

#include <itkImageToImageFilter.h>
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  #include <itkImageRegionSplitterDirection.h>
#endif
namespace rtk
{
  /** \class ImageToVectorImageFilter
   * \brief Re-writes an image as a vector image
   *
   * Depending on the dimensions of the input and output images, the filter
   * can have two different behaviors:
   *  - if the dimensions match, the channels of the input image are
   * obtained by slicing the last dimension. With an input image of size (X, Y*N),
   * the output is an N-components vector image of size (X,Y)
   *  - if the input image dimension equals the output dimension plus one,
   * the additional dimension of the input image is assumed to contain the channels.
   * With an input image of size (X,Y,N), the output is an N-components
   * vector image of size (X, Y).
   *
   * \author Cyril Mory
   */

template< typename InputImageType, typename OutputImageType>
class ImageToVectorImageFilter : public itk::ImageToImageFilter< InputImageType, OutputImageType >
{
public:
    /** Standard class typedefs. */
    typedef ImageToVectorImageFilter                                    Self;
    typedef itk::ImageToImageFilter< InputImageType, OutputImageType >  Superclass;
    typedef itk::SmartPointer< Self >                                   Pointer;

    typedef typename OutputImageType::RegionType                        OutputImageRegionType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ImageToVectorImageFilter, itk::ImageToImageFilter)

    /** When the input and output dimensions are equal, the filter
     * cannot guess the number of channels. Set/Get methods to
     * pass it */
    itkSetMacro(NumberOfChannels, unsigned int)
    itkGetMacro(NumberOfChannels, unsigned int)

protected:
    ImageToVectorImageFilter();
    ~ImageToVectorImageFilter() {}

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

    unsigned int m_NumberOfChannels;

private:
    ImageToVectorImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkImageToVectorImageFilter.hxx"
#endif

#endif
