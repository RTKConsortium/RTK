/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkDrawGeometricPhantomImageFilter_h
#define rtkDrawGeometricPhantomImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkAddImageFilter.h>
#include "rtkGeometricPhantom.h"

namespace rtk
{

/** \class DrawGeometricPhantomImageFilter
 * \brief Draws a GeometricPhantom in a 3D image
 *
 * \test rtkprojectgeometricphantomtest.cxx, rtkforbildtest.cxx
 *
 * \author Marc Vila, Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT DrawGeometricPhantomImageFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DrawGeometricPhantomImageFilter);

  /** Standard class type alias. */
  using Self = DrawGeometricPhantomImageFilter;
  using Superclass = itk::InPlaceImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using GeometricPhantomConstPointer = GeometricPhantom::ConstPointer;
  using StringType = std::string;
  using VectorType = ConvexShape::VectorType;
  using RotationMatrixType = ConvexShape::RotationMatrixType;
  using ScalarType = ConvexShape::ScalarType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(DrawGeometricPhantomImageFilter);
#else
  itkTypeMacro(DrawGeometricPhantomImageFilter, itk::InPlaceImageFilter);
#endif

  /** Get / Set the object pointer to the geometry. */
  itkGetConstObjectMacro(GeometricPhantom, GeometricPhantom);
  itkSetConstObjectMacro(GeometricPhantom, GeometricPhantom);

  /** Get/Set the phantom file path. */
  itkSetMacro(ConfigFile, StringType);
  itkGetMacro(ConfigFile, StringType);

  /** Multiplicative scaling factor along each 3D component. */
  itkSetMacro(PhantomScale, VectorType);
  itkGetMacro(PhantomScale, VectorType);

  /** Set isotropic scaling factor. */
  virtual void
  SetPhantomScale(const ScalarType _arg)
  {
    SetPhantomScale(VectorType(_arg));
  }

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

  /** Add clipping plane to the object. The plane is defined by the equation
   * dir * (x,y,z)' + pos = 0. */
  void
  AddClipPlane(const VectorType & dir, const ScalarType & pos);
  void
  SetClipPlanes(const std::vector<VectorType> & dir, const std::vector<ScalarType> & pos);

protected:
  DrawGeometricPhantomImageFilter();
  ~DrawGeometricPhantomImageFilter() override = default;

  void
  GenerateData() override;

private:
  GeometricPhantomConstPointer m_GeometricPhantom;
  StringType                   m_ConfigFile;
  VectorType                   m_PhantomScale{ 1. };
  VectorType                   m_OriginOffset{ 0. };
  bool                         m_IsForbildConfigFile{ false };
  RotationMatrixType           m_RotationMatrix;
  std::vector<VectorType>      m_PlaneDirections;
  std::vector<ScalarType>      m_PlanePositions;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkDrawGeometricPhantomImageFilter.hxx"
#endif

#endif
