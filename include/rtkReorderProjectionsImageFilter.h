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
 * \brief Sorts projections and other inputs by ascending phase
 *
 * \test
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT ReorderProjectionsImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ReorderProjectionsImageFilter Self;

  typedef itk::ImageToImageFilter<TInputImage, TOutputImage>  Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                             InputImageType;
  typedef TOutputImage                            OutputImageType;
  typedef typename OutputImageType::RegionType    OutputImageRegionType;

  typedef ThreeDCircularProjectionGeometry        GeometryType;
  typedef GeometryType::Pointer                   GeometryPointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ReorderProjectionsImageFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(OutputGeometry, GeometryPointer);
  itkSetMacro(InputGeometry, GeometryPointer);

  /** Set the input signal */
  void SetInputSignal(const std::vector<double> signal);
  std::vector<double> GetOutputSignal();

protected:
  ReorderProjectionsImageFilter();

  ~ReorderProjectionsImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

  // Iterative filters do not need padding
  bool m_PadOnTruncatedSide;

private:
  ReorderProjectionsImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);               //purposely not implemented

  /** RTK geometry objects */
  GeometryPointer m_InputGeometry;
  GeometryPointer m_OutputGeometry;

  /** Input and output signal vectors */
  std::vector<double>   m_InputSignal;
  std::vector<double>   m_OutputSignal;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkReorderProjectionsImageFilter.hxx"
#endif

#endif
