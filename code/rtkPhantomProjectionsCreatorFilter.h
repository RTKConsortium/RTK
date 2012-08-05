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

#ifndef __rtkPhantomProjectionsCreatorFilter_h
#define __rtkPhantomProjectionsCreatorFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkRayQuadricIntersectionImageFilter.h"
#include "rtkGeometricPhantomFileReader.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"

#include <vector>

namespace rtk
{

/** \class PhantomProjectionsCreatorFilter
 * \brief  Computes intersection between source rays and ellipsoids
 *
 * Computes intersection between source rays and ellipsoids,
 * in order to create the projections of a specific phantom which is
 * specified in a configuration file following the convention of
 * http://www.slaney.org/pct/pct-errata.html
 *
 * \author Marc Vila
 *
 * \ingroup RayEllipsoidIntersectionImageFilter
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT PhantomProjectionsCreatorFilter :
  public RayEllipsoidIntersectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef PhantomProjectionsCreatorFilter                                       Self;
  typedef RayEllipsoidIntersectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                       Pointer;
  typedef itk::SmartPointer<const Self>                                 ConstPointer;
  typedef typename TOutputImage::RegionType               OutputImageRegionType;
  typedef typename TOutputImage::Superclass::ConstPointer OutputImageBaseConstPointer;

  typedef float OutputPixelType;

  typedef itk::Image< OutputPixelType, 3 >                                           OutputImageType;
  typedef rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType> REIType;
  typedef std::vector<double>                                                        VectorType;
  typedef std::string                                                                StringType;
  typedef std::vector< std::vector<double> >                                         VectorOfVectorType;
  typedef rtk::GeometricPhantomFileReader                                            CFRType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(PhantomProjectionsCreatorFilter, RayEllipsoidIntersectionImageFilter);

  /** Get/Set Number of Figures.*/
  itkSetMacro(ConfigFile, StringType);
  itkGetMacro(ConfigFile, StringType);

  rtkSetMacro(Fig, VectorOfVectorType);
  rtkGetMacro(Fig, VectorOfVectorType);

protected:
  PhantomProjectionsCreatorFilter() {}
  virtual ~PhantomProjectionsCreatorFilter() {};

  virtual void GenerateData();

  /** Translate user parameteres to quadric parameters.
   * A call to this function will assume modification of the function.*/


private:
  PhantomProjectionsCreatorFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  VectorOfVectorType       m_Fig;
  StringType               m_ConfigFile;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkPhantomProjectionsCreatorFilter.txx"
#endif

#endif
