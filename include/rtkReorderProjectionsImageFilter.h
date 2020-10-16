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

#ifndef rtkReorderProjectionsImageFilter_h
#define rtkReorderProjectionsImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class ReorderProjectionsImageFilter
 * \brief Sorts or shuffle projections and geometry inputs
 *
 * This filter permutes projections and geometry with the same permutation.
 * The permutation is either the one that sorts projections by ascending phase,
 * so that the ones with the same phase can be forward and back projected together
 * (which is faster than one-by-one), or it is a random shuffle, useful for subset
 * processings.
 *
 * \test
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ImageToImageFilter
 */
template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT ReorderProjectionsImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ReorderProjectionsImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ReorderProjectionsImageFilter);
#endif

  /** Standard class type alias. */
  using Self = ReorderProjectionsImageFilter;

  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using OutputImageRegionType = typename OutputImageType::RegionType;
  using PermutationType = enum { NONE = 0, SORT = 1, SHUFFLE = 2 };

  using GeometryType = ThreeDCircularProjectionGeometry;
  using GeometryPointer = GeometryType::Pointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ReorderProjectionsImageFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetModifiableObjectMacro(OutputGeometry, GeometryType);
  itkSetObjectMacro(InputGeometry, GeometryType);

  /** Get / Set the kind of permutation requested */
  itkGetMacro(Permutation, PermutationType);
  itkSetMacro(Permutation, PermutationType);

  /** Set the input signal */
  void
  SetInputSignal(const std::vector<double> signal);
  std::vector<double>
  GetOutputSignal();

protected:
  ReorderProjectionsImageFilter();

  ~ReorderProjectionsImageFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() ITKv5_CONST override;

  void
  GenerateData() override;

private:
  /** RTK geometry objects */
  GeometryPointer m_InputGeometry;
  GeometryPointer m_OutputGeometry;

  /** Input and output signal vectors */
  std::vector<double> m_InputSignal;
  std::vector<double> m_OutputSignal;

  /** Permutation type */
  PermutationType m_Permutation;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkReorderProjectionsImageFilter.hxx"
#endif

#endif
