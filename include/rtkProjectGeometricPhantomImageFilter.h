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
 * \brief Analytical projection a GeometricPhantom
 *
 * \test rtkprojectgeometricphantomtest.cxx, rtkforbildtest.cxx
 *
 * \author Marc Vila, Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ProjectGeometricPhantomImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class type alias. */
  using Self = ProjectGeometricPhantomImageFilter;
  using Superclass = itk::InPlaceImageFilter<TInputImage,TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  using GeometryConstPointer = typename GeometryType::ConstPointer;
  using GeometricPhantomConstPointer = GeometricPhantom::ConstPointer;
  using StringType = std::string;
  using VectorType = ConvexShape::VectorType;
  using RotationMatrixType = ConvexShape::RotationMatrixType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ProjectGeometricPhantomImageFilter, itk::InPlaceImageFilter);

  /** Get / Set the object pointer to the geometry. */
  itkGetConstObjectMacro(GeometricPhantom, GeometricPhantom);
  itkSetConstObjectMacro(GeometricPhantom, GeometricPhantom);

  /** Get / Set the object pointer to projection geometry */
  itkGetConstObjectMacro(Geometry, GeometryType);
  itkSetConstObjectMacro(Geometry, GeometryType);

  /** Get/Set the phantom file path. */
  itkSetMacro(ConfigFile, StringType);
  itkGetMacro(ConfigFile, StringType);

  /** Multiplicative scaling factor along each 3D component. */
  itkSetMacro(PhantomScale, VectorType);
  itkGetMacro(PhantomScale, VectorType);

  /** Get / Set the spatial offset of the phantom relative to its center. The
   * default value is (0, 0, 0). */
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
  ~ProjectGeometricPhantomImageFilter() override {}

  void GenerateData() override;

private:
  ProjectGeometricPhantomImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented

  GeometricPhantomConstPointer m_GeometricPhantom;
  GeometryConstPointer         m_Geometry;
  StringType                   m_ConfigFile;
  VectorType                   m_PhantomScale;
  VectorType                   m_OriginOffset;
  bool                         m_IsForbildConfigFile;
  RotationMatrixType           m_RotationMatrix;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkProjectGeometricPhantomImageFilter.hxx"
#endif

#endif
