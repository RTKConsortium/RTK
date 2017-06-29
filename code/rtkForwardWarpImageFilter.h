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

#ifndef rtkForwardWarpImageFilter_h
#define rtkForwardWarpImageFilter_h

#include <itkWarpImageFilter.h>

#include "rtkMacro.h"

namespace rtk
{

/** \class ForwardWarpImageFilter
 * \brief Warps an image using splat instead of interpolation
 *
 * Deforms an image using a Displacement Vector Field. Adjoint operator
 * of the itkWarpImageFilter
 *
 * \test rtkfourdroostertest
 *
 * \author Cyril Mory
 */
template <class TInputImage,
          class TOutputImage = TInputImage,
          class TDVF = itk::Image< itk::CovariantVector<typename TInputImage::PixelType,
                                                        TInputImage::ImageDimension >,
                                   TInputImage::ImageDimension > >
class ForwardWarpImageFilter :
  public itk::WarpImageFilter< TInputImage, TOutputImage, TDVF >
{
public:
  /** Standard class typedefs. */
  typedef ForwardWarpImageFilter                                  Self;
  typedef itk::WarpImageFilter<TInputImage, TOutputImage, TDVF>   Superclass;
  typedef itk::SmartPointer<Self>                                 Pointer;
  typedef itk::SmartPointer<const Self>                           ConstPointer;

  /** Displacement type */
  typedef TDVF                                      DisplacementFieldType;
  typedef typename DisplacementFieldType::Pointer   DisplacementFieldPointer;
  typedef typename DisplacementFieldType::PixelType DisplacementType;

  /** Point type */
  typedef double                                                   CoordRepType;
  typedef itk::Point< CoordRepType, itkGetStaticConstMacro(ImageDimension) > PointType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForwardWarpImageFilter, Superclass)

protected:
  ForwardWarpImageFilter();
  ~ForwardWarpImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

  // Redefine stuff that is private in the Superclass
  void Protected_EvaluateDisplacementAtPhysicalPoint(const PointType & p, DisplacementType &output);
  bool                                m_Protected_DefFieldSizeSame;
  typename TOutputImage::IndexType    m_Protected_StartIndex;
  typename TOutputImage::IndexType    m_Protected_EndIndex;

private:
  ForwardWarpImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented
};

} // end namespace rtk

#ifndef rtk_MANUAL_INSTANTIATION
#include "rtkForwardWarpImageFilter.hxx"
#endif

#endif
