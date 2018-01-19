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
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class DrawGeometricPhantomImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DrawGeometricPhantomImageFilter                   Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Convenient typedefs. */
  typedef GeometricPhantom::Pointer       GeometricPhantomPointer;
  typedef std::string                     StringType;
  typedef ConvexShape::VectorType         VectorType;
  typedef ConvexShape::RotationMatrixType RotationMatrixType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawGeometricPhantomImageFilter, itk::InPlaceImageFilter);

  /** Get / Set the object pointer to the geometry. */
  itkGetObjectMacro(GeometricPhantom, GeometricPhantom);
  itkSetObjectMacro(GeometricPhantom, GeometricPhantom);

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
  DrawGeometricPhantomImageFilter();
  ~DrawGeometricPhantomImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

private:
  DrawGeometricPhantomImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented

  GeometricPhantomPointer m_GeometricPhantom;
  StringType              m_ConfigFile;
  VectorType              m_PhantomScale;
  VectorType              m_OriginOffset;
  bool                    m_IsForbildConfigFile;
  RotationMatrixType      m_RotationMatrix;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawGeometricPhantomImageFilter.hxx"
#endif

#endif
