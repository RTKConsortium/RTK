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

#ifndef __rtkProjectGeometricPhantomImageFilter_h
#define __rtkProjectGeometricPhantomImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkRayQuadricIntersectionImageFilter.h"
#include "rtkGeometricPhantomFileReader.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "itkAddImageFilter.h"

#include <vector>

namespace rtk
{

/** \class ProjectGeometricPhantomImageFilter
 * \brief  Computes intersection between source rays and ellipsoids
 *
 * Computes intersection between source rays and ellipsoids,
 * in order to create the projections of a specific phantom which is
 * specified in a configuration file following the convention of
 * http://www.slaney.org/pct/pct-errata.html
 *
 * \test rtkprojectgeometricphantomtest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT ProjectGeometricPhantomImageFilter :
  public RayEllipsoidIntersectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ProjectGeometricPhantomImageFilter                            Self;
  typedef RayEllipsoidIntersectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                       Pointer;
  typedef itk::SmartPointer<const Self>                                 ConstPointer;
  typedef typename TOutputImage::RegionType                             OutputImageRegionType;
  typedef typename TOutputImage::Superclass::ConstPointer               OutputImageBaseConstPointer;

  typedef float OutputPixelType;

  typedef TOutputImage                                                               OutputImageType;
  typedef rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType> REIType;
  typedef rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType>       RBIType;
  typedef itk::AddImageFilter <TOutputImage, TOutputImage, TOutputImage>             AddImageFilterType;
  typedef itk::Vector<double, 3>                                                     VectorType;
  typedef std::string                                                                StringType;
  typedef std::vector< std::vector<double> >                                         VectorOfVectorType;
  typedef rtk::GeometricPhantomFileReader                                            CFRType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ProjectGeometricPhantomImageFilter, RayEllipsoidIntersectionImageFilter);

  /** Get/Set Number of Figures.*/
  itkSetMacro(ConfigFile, StringType);
  itkGetMacro(ConfigFile, StringType);

//  struct FigureType
//  {
//    //FigureType():angle(0.),density(0.){};
//    VectorOfVectorType       parameters;
//    std::vector<std::string> figure;
//  };

  virtual VectorOfVectorType GetFig ();
  virtual void SetFig (const VectorOfVectorType _arg);

protected:
  ProjectGeometricPhantomImageFilter() {}
  ~ProjectGeometricPhantomImageFilter() ITK_OVERRIDE {};

  void GenerateData() ITK_OVERRIDE;

  /** Translate user parameteres to quadric parameters.
   * A call to this function will assume modification of the function.*/


private:
  ProjectGeometricPhantomImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  VectorOfVectorType     m_Fig;
  StringType             m_ConfigFile;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkProjectGeometricPhantomImageFilter.hxx"
#endif

#endif
