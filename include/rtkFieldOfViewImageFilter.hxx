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

#ifndef rtkFieldOfViewImageFilter_hxx
#define rtkFieldOfViewImageFilter_hxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#include "lp_lib.h"

#include "rtkProjectionsReader.h"

namespace rtk
{

template<class TInputImage, class TOutputImage>
FieldOfViewImageFilter<TInputImage, TOutputImage>
::FieldOfViewImageFilter():
  m_Geometry(ITK_NULLPTR),
  m_Mask(false),
  m_Radius(-1),
  m_CenterX(0.),
  m_CenterZ(0.),
  m_DisplacedDetector(false)
{
}

template <class TInputImage, class TOutputImage>
bool FieldOfViewImageFilter<TInputImage, TOutputImage>
::ComputeFOVRadius(const FOVRadiusType type, double &x, double &z, double &r)
{
  m_ProjectionsStack->UpdateOutputInformation();
  const unsigned int Dimension = TInputImage::ImageDimension;

  // Compute projection stack indices of corners of inferior X index
  m_ProjectionsStack->UpdateOutputInformation();
  typename TInputImage::IndexType indexCornerInfX1, indexCornerInfX2;
  indexCornerInfX1 = m_ProjectionsStack->GetLargestPossibleRegion().GetIndex();
  indexCornerInfX2 = indexCornerInfX1;
  indexCornerInfX2[1] += m_ProjectionsStack->GetLargestPossibleRegion().GetSize()[1]-1;

  // Compute projection stack indices of corners of superior X index
  typename TInputImage::IndexType indexCornerSupX1, indexCornerSupX2;
  indexCornerSupX1 = indexCornerInfX1;
  indexCornerSupX1[0] += m_ProjectionsStack->GetLargestPossibleRegion().GetSize()[0]-1;
  indexCornerSupX2 = indexCornerInfX2;
  indexCornerSupX2[0] += m_ProjectionsStack->GetLargestPossibleRegion().GetSize()[0]-1;

  // To physical coordinates
  typename TInputImage::PointType cornerInfX1, cornerInfX2, cornerSupX1, cornerSupX2;
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCornerInfX1, cornerInfX1);
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCornerInfX2, cornerInfX2);
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCornerSupX1, cornerSupX1);
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCornerSupX2, cornerSupX2);

  // Build model for lpsolve with 3 variables: x, z and r
  const int Ncol = 3;
  lprec *lp = make_lp(0, Ncol);
  if(lp == ITK_NULLPTR)
    itkExceptionMacro(<< "Couldn't construct 2 new models for the simplex solver");

  // Objective: maximize r
  if(!set_obj(lp, 3, 1.))
    itkExceptionMacro(<< "Couldn't set objective in lpsolve");
  set_maxim(lp);

  set_add_rowmode(lp, TRUE);  // makes building the model faster if it is done rows by row

  int colno[Ncol] = {1, 2, 3};
  REAL row[Ncol];
  for(unsigned int iProj=0; iProj<m_Geometry->GetGantryAngles().size(); iProj++)
    {
    if( m_Geometry->GetSourceToDetectorDistances()[iProj] == 0. )
      itkExceptionMacro(<< "FIXME: parallel case is not handled");

    typename GeometryType::HomogeneousVectorType sourcePosition;
    sourcePosition = m_Geometry->GetSourcePosition(iProj);

    typename GeometryType::ThreeDHomogeneousMatrixType matrix;
    matrix =  m_Geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix();

    // Compute point coordinate in volume depending on projection index
    typename TInputImage::PointType cornerInfX1t, cornerInfX2t, cornerSupX1t, cornerSupX2t;
    for(unsigned int i=0; i<Dimension; i++)
      {
      cornerInfX1t[i] = matrix[i][Dimension];
      cornerInfX2t[i] = matrix[i][Dimension];
      cornerSupX1t[i] = matrix[i][Dimension];
      cornerSupX2t[i] = matrix[i][Dimension];
      for(unsigned int j=0; j<Dimension; j++)
        {
        cornerInfX1t[i] += matrix[i][j] * cornerInfX1[j];
        cornerInfX2t[i] += matrix[i][j] * cornerInfX2[j];
        cornerSupX1t[i] += matrix[i][j] * cornerSupX1[j];
        cornerSupX2t[i] += matrix[i][j] * cornerSupX2[j];
        }
      }

    // Compute the equation of a line of the ax+by=c
    // http://en.wikipedia.org/wiki/Linear_equation#Two-point_form
    double aInf1 = cornerInfX1t[2] - sourcePosition[2];
    double bInf1 = sourcePosition[0] - cornerInfX1t[0];
    double cInf1 = sourcePosition[0] * cornerInfX1t[2] - cornerInfX1t[0] * sourcePosition[2];
    double aInf2 = cornerInfX2t[2] - sourcePosition[2];
    double bInf2 = sourcePosition[0] - cornerInfX2t[0];
    double cInf2 = sourcePosition[0] * cornerInfX2t[2] - cornerInfX2t[0] * sourcePosition[2];
    double aSup1 = cornerSupX1t[2] - sourcePosition[2];
    double bSup1 = sourcePosition[0] - cornerSupX1t[0];
    double cSup1 = sourcePosition[0] * cornerSupX1t[2] - cornerSupX1t[0] * sourcePosition[2];
    double aSup2 = cornerSupX2t[2] - sourcePosition[2];
    double bSup2 = sourcePosition[0] - cornerSupX2t[0];
    double cSup2 = sourcePosition[0] * cornerSupX2t[2] - cornerSupX2t[0] * sourcePosition[2];

    // Then compute the coefficient in front of r as suggested in
    // http://www.ifor.math.ethz.ch/teaching/lectures/intro_ss11/Exercises/solutionEx11-12.pdf
    double dInf1 = std::sqrt(aInf1*aInf1 + bInf1*bInf1);
    double dInf2 = std::sqrt(aInf2*aInf2 + bInf2*bInf2);
    double dSup1 = std::sqrt(aSup1*aSup1 + bSup1*bSup1);
    double dSup2 = std::sqrt(aSup2*aSup2 + bSup2*bSup2);

    // Check on corners
    if( aInf1*cornerSupX1t[0] + bInf1*cornerSupX1t[2] >= cInf1 &&
        aInf2*cornerSupX2t[0] + bInf2*cornerSupX2t[2] >= cInf2 )
      {
      aInf1 *= -1.; bInf1 *= -1.; cInf1 *= -1.;
      aInf2 *= -1.; bInf2 *= -1.; cInf2 *= -1.;
      }
    else if( aSup1*cornerInfX1t[0] + bSup1*cornerInfX1t[2] >= cSup1 &&
             aSup2*cornerInfX2t[0] + bSup2*cornerInfX2t[2] >= cSup2 )
      {
      aSup1 *= -1.; bSup1 *= -1.; cSup1 *= -1.;
      aSup2 *= -1.; bSup2 *= -1.; cSup2 *= -1.;
      }
    else
      {
      itkExceptionMacro(<< "Error computing the FOV, unhandled detector rotation.");
      }

    // Now add the constraints of the form ax+by+dr<=c
    if(type==RADIUSINF || type==RADIUSBOTH)
      {
      row[0] = aInf1; row[1] = bInf1; row[2] = dInf1;
      if(!add_constraintex(lp, 3, row, colno, LE, cInf1))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
      row[0] = aInf2; row[1] = bInf2; row[2] = dInf2;
      if(!add_constraintex(lp, 3, row, colno, LE, cInf2))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
      }
    if(type==RADIUSSUP || type==RADIUSBOTH)
      {
      row[0] = aSup1; row[1] = bSup1; row[2] = dSup1;
      if(!add_constraintex(lp, 3, row, colno, LE, cSup1))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
      row[0] = aSup2; row[1] = bSup2; row[2] = dSup2;
      if(!add_constraintex(lp, 3, row, colno, LE, cSup2))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
      }
    }

  AddCollimationConstraints(type, lp);

  set_add_rowmode(lp, FALSE); // rowmode should be turned off again when done building the model

  if(!set_unbounded(lp, 1) || !set_unbounded(lp, 2))
    itkExceptionMacro(<< "Couldn't not set center to unbounded for simplex");

  set_verbose(lp, IMPORTANT);

  int ret = solve(lp);
  if(ret)
    {
    delete_lp(lp);
    return false;
    }
  else
    {
    get_variables(lp, row);
    x = row[0];
    z = row[1];
    r = row[2];
    }

  delete_lp(lp);
  return true;
}

template <class TInputImage, class TOutputImage>
void FieldOfViewImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  // The radius of the FOV is computed with linear programming.
  if(m_DisplacedDetector)
    {
    // Two radii are computed and the largest is selected.
    if( !ComputeFOVRadius(RADIUSINF, m_CenterX, m_CenterZ, m_Radius) )
      m_Radius = -1.;

    double x,z,r;
    if( ComputeFOVRadius(RADIUSSUP, x, z, r) && r>m_Radius)
      {
      m_Radius = r;
      m_CenterX = x;
      m_CenterZ = z;
      }
    }
  else
    {
    if(!ComputeFOVRadius(RADIUSBOTH, m_CenterX, m_CenterZ, m_Radius))
      m_Radius = -1.;
    }

  // Compute projection stack indices of corners
  m_ProjectionsStack->UpdateOutputInformation();
  typename TInputImage::IndexType indexCorner1;
  indexCorner1 = m_ProjectionsStack->GetLargestPossibleRegion().GetIndex();

  typename TInputImage::IndexType indexCorner2;
  indexCorner2 = indexCorner1 + m_ProjectionsStack->GetLargestPossibleRegion().GetSize();
  for(unsigned int i=0; i<TInputImage::GetImageDimension(); i++)
    indexCorner2[i] --;

  // To physical coordinates
  typename TInputImage::PointType corner1, corner2;
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCorner1, corner1);
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCorner2, corner2);
  for(unsigned int i=0; i<TInputImage::GetImageDimension(); i++)
    if(corner1[i]>corner2[i])
      std::swap(corner1[i], corner2[i]);

  // Go over projection stack, compute minimum radius and minimum tangent
  m_HatHeightSup = itk::NumericTraits<double>::max();
  m_HatHeightInf = itk::NumericTraits<double>::NonpositiveMin();
  for(unsigned int k=0; k<m_ProjectionsStack->GetLargestPossibleRegion().GetSize(2); k++)
    {
    const double sid = m_Geometry->GetSourceToIsocenterDistances()[k];
    const double sdd = m_Geometry->GetSourceToDetectorDistances()[k];
    double mag = 1.;  // Parallel
    if(sdd!=0.)
      mag = sid/sdd;  // Divergent

    const double projOffsetY = m_Geometry->GetProjectionOffsetsY()[k];
    const double sourceOffsetY = m_Geometry->GetSourceOffsetsY()[k];
    const double heightInf = sourceOffsetY+mag*(corner1[1]+projOffsetY-sourceOffsetY);
    if(heightInf>m_HatHeightInf)
      {
      m_HatHeightInf = heightInf;
      m_HatTangentInf = m_HatHeightInf/sid;
      if(sdd==0.) // Parallel
        m_HatTangentInf = 0.;
      }
    const double heightSup = sourceOffsetY+mag*(corner2[1]+projOffsetY-sourceOffsetY);
    if(heightSup<m_HatHeightSup)
      {
      m_HatHeightSup = heightSup;
      m_HatTangentSup = m_HatHeightSup/sid;
      if(sdd==0.) // Parallel
        m_HatTangentSup = 0.;
      }
    }
}

template <class TInputImage, class TOutputImage>
void FieldOfViewImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId )
{
  typename TInputImage::DirectionType d = this->GetInput()->GetDirection();
  if( d[0][0]==1. && d[0][1]==0. && d[0][2]==0. &&
      d[1][0]==0. && d[1][1]==1. && d[1][2]==0. &&
      d[2][0]==0. && d[2][1]==0. && d[2][2]==1.)

    {
    // Prepare point increment (TransformIndexToPhysicalPoint too slow)
    typename TInputImage::PointType pointBase, pointIncrement;
    typename TInputImage::IndexType index = outputRegionForThread.GetIndex();
    this->GetInput()->TransformIndexToPhysicalPoint( index, pointBase );
    for(unsigned int i=0; i<TInputImage::GetImageDimension(); i++)
      index[i]++;
    this->GetInput()->TransformIndexToPhysicalPoint( index, pointIncrement );
    for(unsigned int i=0; i<TInputImage::GetImageDimension(); i++)
      pointIncrement[i] -= pointBase[i];
  
    // Iterators
    typedef itk::ImageRegionConstIterator<TInputImage> InputConstIterator;
    InputConstIterator itIn(this->GetInput(0), outputRegionForThread);
    itIn.GoToBegin();
    typedef itk::ImageRegionIterator<TOutputImage> OutputIterator;
    OutputIterator itOut(this->GetOutput(), outputRegionForThread);
    itOut.GoToBegin();
  
    // Go over output, compute weights and avoid redundant computation
    typename TInputImage::PointType point = pointBase;
    for(unsigned int k=0; k<outputRegionForThread.GetSize(2); k++)
      {
      double zsquare = m_CenterZ - point[2];
      zsquare *= zsquare;
      point[1] = pointBase[1];
      for(unsigned int j=0; j<outputRegionForThread.GetSize(1); j++)
        {
        point[0] = pointBase[0];
        for(unsigned int i=0; i<outputRegionForThread.GetSize(0); i++)
          {
          double xsquare = m_CenterX - point[0];
          xsquare *= xsquare;
          double radius = vcl_sqrt( xsquare + zsquare);
          if ( radius <= m_Radius &&
               radius*m_HatTangentInf >= m_HatHeightInf - point[1] &&
               radius*m_HatTangentSup <= m_HatHeightSup - point[1])
            {
            if(m_Mask)
              itOut.Set(1.0);
            else
              itOut.Set(itIn.Get());
            }
          else
            itOut.Set(0.);
          ++itIn;
          ++itOut;
          point[0] += pointIncrement[0];
          }
        point[1] += pointIncrement[1];
        }
      point[2] += pointIncrement[2];
      }
    }
  else
    {
    typedef itk::ImageRegionConstIteratorWithIndex<TInputImage> InputConstIterator;
    InputConstIterator itIn(this->GetInput(0), outputRegionForThread);

    typedef itk::ImageRegionIterator<TOutputImage> OutputIterator;
    OutputIterator itOut(this->GetOutput(), outputRegionForThread);

    typename TInputImage::PointType point;
    while( !itIn.IsAtEnd() ) 
      {
      this->GetInput()->TransformIndexToPhysicalPoint( itIn.GetIndex(), point );
      double radius = vcl_sqrt(point[0]*point[0] + point[2]*point[2]);
      if ( radius <= m_Radius &&
           point[1] <= m_HatHeightSup - radius*m_HatTangentSup &&
           point[1] >= m_HatHeightInf - radius*m_HatTangentInf )
        {
        if(m_Mask)
          itOut.Set(1.0);
        else
          itOut.Set(itIn.Get());
        }
      else
        itOut.Set(0.);
      ++itIn;
      ++itOut;
      }
    }
}

template <class TInputImage, class TOutputImage>
void
FieldOfViewImageFilter<TInputImage, TOutputImage>
::AddCollimationConstraints(const FOVRadiusType type, _lprec *lp)
{
  const int Ncol = 3;
  int colno[Ncol] = {1, 2, 3};
  REAL row[Ncol];
  for(unsigned int iProj=0; iProj<m_Geometry->GetGantryAngles().size(); iProj++)
    {
    const double X1 = m_Geometry->GetCollimationUInf()[iProj];
    const double X2 = m_Geometry->GetCollimationUSup()[iProj];
    if(X1 == std::numeric_limits<double>::max() &&
       X2 == std::numeric_limits<double>::max())
      {
      continue;
      }
    if(X1 == std::numeric_limits<double>::max() ||
       X2 == std::numeric_limits<double>::max())
      {
      itkWarningMacro("Having only one jaw that is not at the default value is unexpected.")
      }

    //Compute 3D position of jaws
    typedef typename GeometryType::VectorType PointType;
    typename GeometryType::HomogeneousVectorType sourceH = m_Geometry->GetSourcePosition(iProj);
    PointType source(0.);
    source[0] = sourceH[0];
    source[2] = sourceH[2];
    double sourceNorm = source.GetNorm();
    PointType sourceDir = source/sourceNorm;

    PointType v(0.);
    v[1] = 1.;
    PointType u = CrossProduct(v, sourceDir);

    // Compute the equation of a line of the ax+by=c
    // http://en.wikipedia.org/wiki/Linear_equation#Two-point_form
    // Then compute the coefficient in front of r as suggested in
    // http://www.ifor.math.ethz.ch/teaching/lectures/intro_ss11/Exercises/solutionEx11-12.pdf
    PointType inf = u*-1.*X1;
    double aInf = inf[2] - source[2];
    double bInf = source[0] - inf[0];
    double cInf = source[0] * inf[2] - inf[0] * source[2];
    double dInf = std::sqrt(aInf*aInf + bInf*bInf);

    PointType sup = u*X2;
    double aSup = sup[2] - source[2];
    double bSup = source[0] - sup[0];
    double cSup = source[0] * sup[2] - sup[0] * source[2];
    double dSup = std::sqrt(aSup*aSup + bSup*bSup);

    // Check on corners
    if( aInf*sup[0] + bInf*sup[2] >= cInf )
      {
      aInf *= -1.; bInf *= -1.; cInf *= -1.;
      }
    else if( aSup*inf[0] + bSup*inf[2] >= cSup )
      {
      aSup *= -1.; bSup *= -1.; cSup *= -1.;
      }
    else
      {
      itkExceptionMacro(<< "Something's wrong with the jaw handling.");
      }

    // Now add the constraints of the form ax+by+dr<=c
    if(type==RADIUSINF || type==RADIUSBOTH)
      {
      row[0] = aInf; row[1] = bInf; row[2] = dInf;
      if(!add_constraintex(lp, 3, row, colno, LE, cInf))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
      }
    if(type==RADIUSSUP || type==RADIUSBOTH)
      {
      row[0] = aSup; row[1] = bSup; row[2] = dSup;
      if(!add_constraintex(lp, 3, row, colno, LE, cSup))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
      }
    }
}

} // end namespace rtk

#endif // rtkFieldOfViewImageFilter_hxx
