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

#ifndef rtkDrawConeImageFilter_h
#define rtkDrawConeImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkDrawQuadricImageFilter.h"


#include "rtkConfiguration.h"
#include <vector>

namespace rtk
{

/** \class DrawConeImageFilter
 * \brief Draws in a 3D image user defined Cone.
 *
 * \test rtkdrawgeometricphantomtest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage,
          class TOutputImage,
          typename TFunction = itk::Functor::Add2<typename TInputImage::PixelType,
                                                  typename TInputImage::PixelType,
                                                  typename TOutputImage::PixelType>
                                                  >
class ITK_EXPORT DrawConeImageFilter :
public DrawQuadricImageFilter<TInputImage,
                              TOutputImage,
                              DrawQuadricSpatialObject,
                              TFunction
                             >
{
       public:
  /** Standard class typedefs. */
  typedef DrawConeImageFilter                               Self;
  typedef DrawQuadricImageFilter < TInputImage,
                                   TOutputImage,
                                   DrawQuadricSpatialObject,
                                   TFunction >              Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;

  typedef itk::Vector<double,3>                             VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawConeImageFilter, DrawQuadricSpatialObject);

protected:
  DrawConeImageFilter();
  virtual ~DrawConeImageFilter() ITK_OVERRIDE {}

private:
  DrawConeImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);      //purposely not implemented
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawConeImageFilter.hxx"
#endif

#endif
