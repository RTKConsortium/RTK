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

#ifndef rtkProjectionGeometry_h
#define rtkProjectionGeometry_h

#include <itkImageBase.h>

#include <vector>

#include "rtkMacro.h"

namespace rtk
{

/** \class ProjectionGeometry
 * \brief A templated class holding a vector of M x (M+1) matrices
 *
 * This class contains a vector of projection matrices.
 * Each matrix corresponds to a different position of a
 * projector, e.g. a detector and an x-ray source.
 * The class is meant to be specialized for specific geometries.
 *
 * \author Simon Rit
 *
 * \ingroup Geometry
 */
template< unsigned int TDimension = 3 >
class ProjectionGeometry : public itk::DataObject
{
public:
  typedef ProjectionGeometry<TDimension>  Self;
  typedef itk::DataObject                 Superclass;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Convenient typedefs */
  typedef typename itk::ImageBase<TDimension>::SizeType    SizeType;
  typedef typename itk::ImageBase<TDimension>::PointType   PointType;
  typedef typename itk::ImageBase<TDimension>::SpacingType SpacingType;

  typedef typename itk::Matrix< double, TDimension, TDimension+1 > MatrixType;

  /** Get the vector of projection matrices.
   * A projection matrix is a M x (M+1) homogeneous matrix.
   * The multiplication of a M-D point in physical coordinates
   * with the i-th matrix provides the physical coordinate on
   * the i-th projection.
   */
  const std::vector<MatrixType> &GetMatrices(){
    return this->m_Matrices;
  }

  /** Empty the geometry object. */
  virtual void Clear();

protected:
  ProjectionGeometry(){};
  ~ProjectionGeometry() {}

  void PrintSelf( std::ostream& os, itk::Indent indent ) const ITK_OVERRIDE;

  /** Add projection matrix */
  virtual void AddMatrix(const MatrixType &m){
    this->m_Matrices.push_back(m);
    this->Modified();
  }

private:
  ProjectionGeometry(const Self&); //purposely not implemented
  void operator=(const Self&);     //purposely not implemented

  /** Projection matrices */
  std::vector<MatrixType> m_Matrices;
};
}

#include "rtkProjectionGeometry.hxx"

#endif // rtkProjectionGeometry_h
