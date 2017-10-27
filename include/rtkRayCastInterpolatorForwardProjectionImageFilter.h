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

#ifndef rtkRayCastInterpolatorForwardProjectionImageFilter_h
#define rtkRayCastInterpolatorForwardProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"

namespace rtk
{

/** \class RayCastInterpolatorForwardProjectionImageFilter
 * \brief Forward projection using itk RayCastInterpolateFunction
 *
 * Forward projection using itk RayCastInterpolateFunction.
 * RayCastInterpolateFunction does not handle ITK geometry correctly but this
 * is accounted for this class.
 *
 * \test rtkRaycastInterpolatorForwardProjectionTest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup Projector
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayCastInterpolatorForwardProjectionImageFilter :
  public ForwardProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayCastInterpolatorForwardProjectionImageFilter        Self;
  typedef ForwardProjectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                Pointer;
  typedef itk::SmartPointer<const Self>                          ConstPointer;

  /** Useful typedefs. */
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayCastInterpolatorForwardProjectionImageFilter, ForwardProjectionImageFilter);

protected:
  RayCastInterpolatorForwardProjectionImageFilter() {}
  ~RayCastInterpolatorForwardProjectionImageFilter() {}

  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

private:
  RayCastInterpolatorForwardProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayCastInterpolatorForwardProjectionImageFilter.hxx"
#endif

#endif
