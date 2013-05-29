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

#ifndef __rtkDrawCylinderImageFilter_h
#define __rtkDrawCylinderImageFilter_h

#include <itkInPlaceImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include "rtkConfiguration.h"
#include <vector>

namespace rtk
{

/** \class DrawCylinderImageFilter
 * \brief Draws in a 3D image user defined Cylinder.
 *
 * \test rtkdrawgeometricphantomtest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT DrawCylinderImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DrawCylinderImageFilter                           Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;

  typedef std::vector<double>                               VectorType;

  typedef rtk::ConvertEllipsoidToQuadricParametersFunction  EQPFunctionType;
  struct FigureType
  {
    FigureType():angle(0.),attenuation(0.){};
    VectorType semiprincipalaxis;
    VectorType center;
    double     angle;
    double     attenuation;
  };
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawCylinderImageFilter, InPlaceImageFilter);

  /** Multiplicative Scaling factor for the phantom parameters described in
   * http://www.slaney.org/pct/pct-errata.html. */
  itkSetMacro(Attenuation, double);
  itkGetMacro(Attenuation, double);

  itkSetMacro(Angle, double);
  itkGetMacro(Angle, double);

  rtkSetStdVectorMacro(Axis, VectorType );
  rtkGetStdVectorMacro(Axis, VectorType );

  rtkSetStdVectorMacro(Center, VectorType );
  rtkGetStdVectorMacro(Center, VectorType );

protected:
  DrawCylinderImageFilter();
  virtual ~DrawCylinderImageFilter() {};

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

private:
  DrawCylinderImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  std::vector< double > m_Axis;
  std::vector< double > m_Center;
  double                m_Attenuation;
  double                m_Angle;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawCylinderImageFilter.txx"
#endif

#endif
