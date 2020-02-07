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
#ifndef rtkMultiplyByVectorImageFilter_h
#define rtkMultiplyByVectorImageFilter_h

#include <itkImageToImageFilter.h>

namespace rtk
{
/** \class MultiplyByVectorImageFilter
 * \brief Multiplies each (n-1) dimension image by the corresponding
 * element in a vector
 *
 * This filter takes in input a n-D image and a vector. It multiplies
 * each (n-1) dimension image by the corresponding
 * element in the vector. The image's size along the last dimension
 * must be equal to vector's size.
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 *
 */
template <class TInputImage>
class MultiplyByVectorImageFilter : public itk::ImageToImageFilter<TInputImage, TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(MultiplyByVectorImageFilter);

  /** Standard class type alias. */
  using Self = MultiplyByVectorImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TInputImage>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiplyByVectorImageFilter, itk::ImageToImageFilter);

  /** The image containing the weights applied to the temporal components */
  void
  SetVector(std::vector<float> vect);

protected:
  MultiplyByVectorImageFilter();
  ~MultiplyByVectorImageFilter() override = default;

  /** Does the real work. */
  void
  DynamicThreadedGenerateData(const typename TInputImage::RegionType & outputRegionForThread) override;

private:
  std::vector<float> m_Vector;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkMultiplyByVectorImageFilter.hxx"
#endif

#endif
