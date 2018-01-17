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

#ifndef rtkProjectGeometricPhantomImageFilter_h
#define rtkProjectGeometricPhantomImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkAddImageFilter.h>
#include "rtkGeometricPhantom.h"
#include "rtkThreeDCircularProjectionGeometry.h"

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
class ProjectGeometricPhantomImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ProjectGeometricPhantomImageFilter                Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Convenient typedefs. */
  typedef rtk::ThreeDCircularProjectionGeometry           GeometryType;
  typedef typename GeometryType::Pointer                  GeometryPointer;
  typedef GeometricPhantom::Pointer                       GeometricPhantomPointer;
  typedef std::string                                     StringType;
  typedef ConvexShape::VectorType                         VectorType;
  typedef ConvexShape::RotationMatrixType                 RotationMatrixType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ProjectGeometricPhantomImageFilter, itk::InPlaceImageFilter);

  /** Get / Set the object pointer to the geometry. */
  itkGetObjectMacro(GeometricPhantom, GeometricPhantom);
  itkSetObjectMacro(GeometricPhantom, GeometricPhantom);

  /** Get / Set the object pointer to projection geometry */
  itkGetObjectMacro(Geometry, GeometryType);
  itkSetObjectMacro(Geometry, GeometryType);

  /** Get/Set Number of Figures.*/
  itkSetMacro(ConfigFile, StringType);
  itkGetMacro(ConfigFile, StringType);

  /** Multiplicative Scaling factor for the phantom described ConfigFile. */
  itkSetMacro(PhantomScale, VectorType);
  itkGetMacro(PhantomScale, VectorType);

  /** Get / Set the spatial position of the phantom Shepp Logan phantom relative to its
   * center. The default value is (0, 0, 0). */
  itkSetMacro(OriginOffset, VectorType);
  itkGetMacro(OriginOffset, VectorType);

  /** Interpret config file as Forbild file (see
   * http://www.imp.uni-erlangen.de/phantoms/). */
  itkSetMacro(IsForbildConfigFile, bool);
  itkGetConstMacro(IsForbildConfigFile, bool);
  itkBooleanMacro(IsForbildConfigFile);

  /** Get / Set a rotation matrix for the phantom. Default is identity. */
  itkSetMacro(RotationMatrix, RotationMatrixType);
  itkGetMacro(RotationMatrix, RotationMatrixType);

protected:
  ProjectGeometricPhantomImageFilter();
  ~ProjectGeometricPhantomImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

private:
  ProjectGeometricPhantomImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented

  GeometricPhantomPointer m_GeometricPhantom;
  GeometryPointer         m_Geometry;
  StringType              m_ConfigFile;
  VectorType              m_PhantomScale;
  VectorType              m_OriginOffset;
  bool                    m_IsForbildConfigFile;
  RotationMatrixType      m_RotationMatrix;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkProjectGeometricPhantomImageFilter.hxx"
#endif

#endif
