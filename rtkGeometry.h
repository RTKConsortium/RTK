#ifndef RTKGEOMETRY_H
#define RTKGEOMETRY_H

#include "rtkMacro.h"

#include <itkImageBase.h>

#include <vector>

namespace rtk
{
template< unsigned int TDimension = 3 >
class Geometry:public itk::DataObject
{
public:
  typedef Geometry<TDimension>            Self;
  typedef itk::DataObject                 Superclass;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Convenient typedefs */
  typedef typename itk::ImageBase<TDimension>::SizeType            SizeType;
  typedef typename itk::ImageBase<TDimension>::PointType           PointType;
  typedef typename itk::ImageBase<TDimension>::SpacingType         SpacingType;

  typedef typename itk::Matrix< double, TDimension, TDimension+1 > MatrixType;

  /** Get the vector of matrices */
  const std::vector<MatrixType> &GetMatrices(){
    return this->m_Matrices;
  }

protected:

  Geometry(){};
  virtual ~Geometry(){};

  virtual void PrintSelf( std::ostream& os, itk::Indent indent ) const;

  /** Add projection matrix */
  virtual void AddMatrix(const MatrixType &m){
    this->m_Matrices.push_back(m);
    this->Modified();
  }

private:
  Geometry(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

private:
  /** Projection matrices */
  std::vector<MatrixType> m_Matrices;
};
}

#include "rtkGeometry.txx"

#endif // RTKGEOMETRY_H
