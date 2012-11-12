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

#ifndef __rtkDrawSphereFilter_h
#define __rtkDrawSphereFilter_h

#include <itkInPlaceImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"

namespace rtk
{

/** \class DrawSphereFilter
 * \brief Draws in a 3D image user defined sphere.
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT DrawSphereFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DrawSphereFilter                                  Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;

  typedef std::vector<double>                               VectorType;
  typedef std::string                                       StringType;

  typedef rtk::ConvertEllipsoidToQuadricParametersFunction  EQPFunctionType;
  struct figure
  {
    VectorType semiprincipalaxis;
    VectorType center;
    double     angle;
    double     attenuation;
  };
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawSphereFilter, InPlaceImageFilter);

  /** Multiplicative Scaling factor for the phantom parameters described in
   * http://www.slaney.org/pct/pct-errata.html. */
  itkSetMacro(SphereScale, double);
  itkGetMacro(SphereScale, double);

protected:
  DrawSphereFilter();
  virtual ~DrawSphereFilter() {};

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

private:
  DrawSphereFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  double m_SphereScale;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawSphereFilter.txx"
#endif

#endif
