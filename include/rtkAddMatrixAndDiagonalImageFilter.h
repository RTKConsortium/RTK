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
#ifndef rtkAddMatrixAndDiagonalImageFilter_h
#define rtkAddMatrixAndDiagonalImageFilter_h

#include "itkImageToImageFilter.h"
#include "rtkMacro.h"

namespace rtk
{
  /** \class AddMatrixAndDiagonalImageFilter
   * \brief For each vector-valued pixel, adds a vector to the diagonal of a matrix
   *
   * This filter takes in input an image of vectors of length n (input 1)
   * and an image of vectors of length n*n (input 2). The vectors in input 2
   * are used as n*n matrices, and those in input 1 are assumed to be a compact
   * representation of diagonal matrices of size n*n (thus with only n non-null
   * values).
   *
   * \author Cyril Mory
   *
   */
template< class TDiagonal,
          class TMatrix = itk::Image<itk::Vector<typename TDiagonal::PixelType::ValueType, TDiagonal::PixelType::Dimension * TDiagonal::PixelType::Dimension>, TDiagonal::ImageDimension > >
class AddMatrixAndDiagonalImageFilter : public itk::ImageToImageFilter<TMatrix, TMatrix>
{
public:
    /** Standard class typedefs. */
    typedef AddMatrixAndDiagonalImageFilter               Self;
    typedef itk::ImageToImageFilter<TMatrix, TMatrix>     Superclass;
    typedef itk::SmartPointer< Self >                     Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(AddMatrixAndDiagonalImageFilter, itk::ImageToImageFilter)

    /** Convenient parameters extracted from template types */
    itkStaticConstMacro(nChannels, unsigned int, TDiagonal::PixelType::Dimension);

    /** Convenient typedef */
    typedef typename TDiagonal::PixelType::ValueType dataType;

    /** Set methods for all inputs, since they have different types */
    void SetInputDiagonal(const TDiagonal* gradient);
    void SetInputMatrix(const TMatrix* hessian);

protected:
    AddMatrixAndDiagonalImageFilter();
    ~AddMatrixAndDiagonalImageFilter() {}

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Does the real work. */
#if ITK_VERSION_MAJOR<5
    void ThreadedGenerateData(const typename TDiagonal::RegionType& outputRegionForThread, itk::ThreadIdType threadId) ITK_OVERRIDE;
#else
    void DynamicThreadedGenerateData(const typename TDiagonal::RegionType& outputRegionForThread) ITK_OVERRIDE;
#endif

    /** Getters for the inputs */
    typename TDiagonal::ConstPointer GetInputDiagonal();
    typename TMatrix::ConstPointer GetInputMatrix();

private:
    AddMatrixAndDiagonalImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkAddMatrixAndDiagonalImageFilter.hxx"
#endif

#endif
