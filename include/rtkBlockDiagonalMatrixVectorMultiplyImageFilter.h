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
#ifndef rtkBlockDiagonalMatrixVectorMultiplyImageFilter_h
#define rtkBlockDiagonalMatrixVectorMultiplyImageFilter_h

#include "itkImageToImageFilter.h"
#include "rtkMacro.h"

namespace rtk
{
  /** \class BlockDiagonalMatrixVectorMultiplyImageFilter
   * \brief Multiplies matrix by vector
   *
   * \author Cyril Mory
   *
   */
template< class TVectorImage,
          class TMatrixImage = itk::Image<itk::Vector<typename TVectorImage::PixelType::ValueType, TVectorImage::PixelType::Dimension * TVectorImage::PixelType::Dimension>, TVectorImage::ImageDimension > >
class BlockDiagonalMatrixVectorMultiplyImageFilter : public itk::ImageToImageFilter<TVectorImage, TVectorImage>
{
public:
    /** Standard class typedefs. */
    typedef BlockDiagonalMatrixVectorMultiplyImageFilter                    Self;
    typedef itk::ImageToImageFilter<TVectorImage, TVectorImage> Superclass;
    typedef itk::SmartPointer< Self >                     Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(BlockDiagonalMatrixVectorMultiplyImageFilter, itk::ImageToImageFilter)

    /** Convenient parameters extracted from template types */
    itkStaticConstMacro(nChannels, unsigned int, TVectorImage::PixelType::Dimension);

    /** Convenient typedef */
    typedef typename TVectorImage::PixelType::ValueType dataType;

    /** Set methods for all inputs, since they have different types */
    void SetInput1(const TVectorImage* vector);
    void SetInput2(const TMatrixImage* matrix);

protected:
    BlockDiagonalMatrixVectorMultiplyImageFilter();
    ~BlockDiagonalMatrixVectorMultiplyImageFilter() {}

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Does the real work. */
#if ITK_VERSION_MAJOR<5
    void ThreadedGenerateData(const typename TVectorImage::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;
#else
    void DynamicThreadedGenerateData(const typename TVectorImage::RegionType& outputRegionForThread) ITK_OVERRIDE;
#endif

    /** Getters for the inputs */
    typename TVectorImage::ConstPointer GetInput1();
    typename TMatrixImage::ConstPointer GetInput2();

private:
    BlockDiagonalMatrixVectorMultiplyImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkBlockDiagonalMatrixVectorMultiplyImageFilter.hxx"
#endif

#endif
