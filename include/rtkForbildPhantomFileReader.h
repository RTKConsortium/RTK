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

#ifndef rtkForbildPhantomFileReader_h
#define rtkForbildPhantomFileReader_h

#include <itkLightProcessObject.h>
#include "rtkGeometricPhantom.h"

namespace rtk
{

/** \class ForbildPhantomFileReader
 * \brief Reads GeometricPhantom from a Forbild file
 *
 * See http://www.imp.uni-erlangen.de/phantoms/. In addition to the original
 * parameters, the parameter "union=N", allows defining the union with a previous
 * shape. For example, union=-1 will create a union with the previous shape.
 *
 * \test rtkforbildtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
class RTK_EXPORT ForbildPhantomFileReader : public itk::LightProcessObject
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ForbildPhantomFileReader);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ForbildPhantomFileReader);
#endif

  /** Standard class type alias. */
  using Self = ForbildPhantomFileReader;
  using Superclass = itk::Object;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  static constexpr unsigned int Dimension = ConvexShape::Dimension;
  using GeometricPhantomPointer = GeometricPhantom::Pointer;
  using ScalarType = ConvexShape::ScalarType;
  using PointType = ConvexShape::PointType;
  using VectorType = ConvexShape::VectorType;
  using RotationMatrixType = ConvexShape::RotationMatrixType;
  using ConvexShapeVectorType = GeometricPhantom::ConvexShapeVector;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForbildPhantomFileReader, itk::LightProcessObject);

  /** Get / Set the object pointer to geometric phantom. */
  itkGetModifiableObjectMacro(GeometricPhantom, GeometricPhantom);
  itkSetObjectMacro(GeometricPhantom, GeometricPhantom);

  /** Get/Set path to phantom definition file. */
  itkGetStringMacro(Filename);
  itkSetStringMacro(Filename);

  /** do the actual parsing of the input file */
  virtual void
  GenerateOutputInformation();

protected:
  /// Constructor
  ForbildPhantomFileReader() = default;

  /// Destructor
  ~ForbildPhantomFileReader() override = default;

  void
  CreateForbildSphere(const std::string & s);
  void
  CreateForbildBox(const std::string & s);
  void
  CreateForbildCylinder(const std::string & s, const std::string & fig);
  void
  CreateForbildElliptCyl(const std::string & s, const std::string & fig);
  void
  CreateForbildEllipsoid(const std::string & s, const std::string & fig);
  void
  CreateForbildCone(const std::string & s, const std::string & fig);
  void
  CreateForbildTetrahedron(const std::string & s);
  RotationMatrixType
  ComputeRotationMatrixBetweenVectors(const VectorType & source, const VectorType & dest) const;

  bool
  FindParameterInString(const std::string & name, const std::string & s, ScalarType & param);
  bool
  FindVectorInString(const std::string & name, const std::string & s, VectorType & vec);
  void
  FindClipPlanes(const std::string & s);
  void
  FindUnions(const std::string & s);

private:
  GeometricPhantomPointer m_GeometricPhantom;
  std::string             m_Filename;
  PointType               m_Center;
  ConvexShape::Pointer    m_ConvexShape;
  ConvexShapeVectorType   m_Unions;
};

} // end namespace rtk

#endif
