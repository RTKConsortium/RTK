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

#ifndef rtkFieldOfViewImageFilter_hxx
#define rtkFieldOfViewImageFilter_hxx

#include "math.h"


#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#include "lp_lib.h"

#include "rtkProjectionsReader.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
FieldOfViewImageFilter<TInputImage, TOutputImage>::FieldOfViewImageFilter() = default;

template <class TInputImage, class TOutputImage>
bool
FieldOfViewImageFilter<TInputImage, TOutputImage>::ComputeFOVRadius(const FOVRadiusType type,
                                                                    double &            x,
                                                                    double &            z,
                                                                    double &            r)
{
  m_ProjectionsStack->UpdateOutputInformation();
  typename TInputImage::SizeType regSize;
  regSize.Fill(1);
  typename TInputImage::RegionType region = m_ProjectionsStack->GetLargestPossibleRegion();
  region.SetSize(regSize);
  auto dumImg = TInputImage::New();
  dumImg->CopyInformation(m_ProjectionsStack);

  // Build model for lpsolve with 3 variables: x, z and r
  constexpr int Ncol = 3;
  lprec *       lp = make_lp(0, Ncol);
  if (lp == nullptr)
    itkExceptionMacro(<< "Couldn't construct 2 new models for the simplex solver");

  // Objective: maximize r
  if (!set_obj(lp, 3, 1.))
    itkExceptionMacro(<< "Couldn't set objective in lpsolve");
  set_maxim(lp);

  set_add_rowmode(lp, TRUE); // makes building the model faster if it is done rows by row

  int  colno[Ncol] = { 1, 2, 3 };
  REAL row[Ncol];
  for (unsigned int iProj = 0; iProj < m_Geometry->GetGantryAngles().size(); iProj++)
  {
    constexpr unsigned int NCORNERS = 4;
    double                 a[NCORNERS];
    double                 b[NCORNERS];
    double                 c[NCORNERS];
    double                 d[NCORNERS];
    using InputRegionIterator = ProjectionsRegionConstIteratorRayBased<TInputImage>;
    InputRegionIterator *                   itIn = nullptr;
    typename InputRegionIterator::PointType corners[NCORNERS];
    for (unsigned int i = 0; i < NCORNERS; i++)
    {
      // Create image iterator with geometry for that particular pixel (== corner)
      typename InputRegionIterator::PointType sourcePosition;
      region.SetIndex(0,
                      m_ProjectionsStack->GetLargestPossibleRegion().GetIndex()[0] +
                        (i / 2) * (m_ProjectionsStack->GetLargestPossibleRegion().GetSize()[0] - 1));
      region.SetIndex(1,
                      m_ProjectionsStack->GetLargestPossibleRegion().GetIndex()[1] +
                        (i % 2) * (m_ProjectionsStack->GetLargestPossibleRegion().GetSize()[1] - 1));
      region.SetIndex(2, iProj);
      dumImg->SetRegions(region);
      dumImg->Allocate();
      itIn = InputRegionIterator::New(dumImg, region, m_Geometry);

      // Compute the equation of a line of the ax+by=c
      // https://en.wikipedia.org/wiki/Linear_equation#Two-point_form
      sourcePosition = itIn->GetSourcePosition();
      corners[i] = itIn->GetPixelPosition();
      delete itIn;

      a[i] = corners[i][2] - sourcePosition[2];
      b[i] = sourcePosition[0] - corners[i][0];
      c[i] = sourcePosition[0] * corners[i][2] - corners[i][0] * sourcePosition[2];

      // Then compute the coefficient in front of r as suggested in
      // http://www.ifor.math.ethz.ch/teaching/lectures/intro_ss11/Exercises/solutionEx11-12.pdf
      d[i] = std::sqrt(a[i] * a[i] + b[i] * b[i]);
    }

    // Check on corners
    if (a[0] * corners[2][0] + b[0] * corners[2][2] >= c[0] && a[1] * corners[3][0] + b[1] * corners[3][2] >= c[1])
    {
      a[0] *= -1.;
      b[0] *= -1.;
      c[0] *= -1.;
      a[1] *= -1.;
      b[1] *= -1.;
      c[1] *= -1.;
    }
    else if (a[2] * corners[0][0] + b[2] * corners[0][2] >= c[2] && a[3] * corners[1][0] + b[3] * corners[1][2] >= c[3])
    {
      a[2] *= -1.;
      b[2] *= -1.;
      c[2] *= -1.;
      a[3] *= -1.;
      b[3] *= -1.;
      c[3] *= -1.;
    }
    else
    {
      itkExceptionMacro(<< "Error computing the FOV, unhandled detector rotation.");
    }

    // Now add the constraints of the form ax+by+dr<=c
    if (type == RADIUSINF || type == RADIUSBOTH)
    {
      row[0] = a[0];
      row[1] = b[0];
      row[2] = d[0];
      if (!add_constraintex(lp, 3, row, colno, LE, c[0]))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
      row[0] = a[1];
      row[1] = b[1];
      row[2] = d[1];
      if (!add_constraintex(lp, 3, row, colno, LE, c[1]))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
    }
    if (type == RADIUSSUP || type == RADIUSBOTH)
    {
      row[0] = a[2];
      row[1] = b[2];
      row[2] = d[2];
      if (!add_constraintex(lp, 3, row, colno, LE, c[2]))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
      row[0] = a[3];
      row[1] = b[3];
      row[2] = d[3];
      if (!add_constraintex(lp, 3, row, colno, LE, c[3]))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
    }
  }

  AddCollimationConstraints(type, lp);

  set_add_rowmode(lp, FALSE); // rowmode should be turned off again when done building the model

  if (!set_unbounded(lp, 1) || !set_unbounded(lp, 2))
    itkExceptionMacro(<< "Couldn't not set center to unbounded for simplex");

  set_verbose(lp, IMPORTANT);

  int ret = solve(lp);
  if (ret)
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
void
FieldOfViewImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class TInputImage, class TOutputImage>
void
FieldOfViewImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  // The radius of the FOV is computed with linear programming.
  if (m_DisplacedDetector)
  {
    // Two radii are computed and the largest is selected.
    if (!ComputeFOVRadius(RADIUSINF, m_CenterX, m_CenterZ, m_Radius))
      m_Radius = -1.;

    double x = NAN, z = NAN, r = NAN;
    if (ComputeFOVRadius(RADIUSSUP, x, z, r) && r > m_Radius)
    {
      m_Radius = r;
      m_CenterX = x;
      m_CenterZ = z;
    }
  }
  else
  {
    if (!ComputeFOVRadius(RADIUSBOTH, m_CenterX, m_CenterZ, m_Radius))
      m_Radius = -1.;
  }

  // Compute projection stack indices of corners
  m_ProjectionsStack->UpdateOutputInformation();
  typename TInputImage::IndexType indexCorner1;
  indexCorner1 = m_ProjectionsStack->GetLargestPossibleRegion().GetIndex();

  typename TInputImage::IndexType indexCorner2;
  indexCorner2 = indexCorner1 + m_ProjectionsStack->GetLargestPossibleRegion().GetSize();
  for (unsigned int i = 0; i < TInputImage::GetImageDimension(); i++)
    indexCorner2[i]--;

  // To physical coordinates
  typename TInputImage::PointType corner1, corner2;
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCorner1, corner1);
  m_ProjectionsStack->TransformIndexToPhysicalPoint(indexCorner2, corner2);
  for (unsigned int i = 0; i < TInputImage::GetImageDimension(); i++)
    if (corner1[i] > corner2[i])
      std::swap(corner1[i], corner2[i]);

  // Go over projection stack, compute minimum radius and minimum tangent
  m_HatHeightSup = itk::NumericTraits<double>::max();
  m_HatHeightInf = itk::NumericTraits<double>::NonpositiveMin();
  for (unsigned int k = 0; k < m_ProjectionsStack->GetLargestPossibleRegion().GetSize(2); k++)
  {
    const double sid = m_Geometry->GetSourceToIsocenterDistances()[k];
    const double sdd = m_Geometry->GetSourceToDetectorDistances()[k];
    double       mag = 1.; // Parallel
    if (sdd != 0.)
      mag = sid / sdd; // Divergent

    const double projOffsetY = m_Geometry->GetProjectionOffsetsY()[k];
    const double sourceOffsetY = m_Geometry->GetSourceOffsetsY()[k];
    const double heightInf = sourceOffsetY + mag * (corner1[1] + projOffsetY - sourceOffsetY);
    if (heightInf > m_HatHeightInf)
    {
      m_HatHeightInf = heightInf;
      m_HatTangentInf = m_HatHeightInf / sid;
      if (sdd == 0.) // Parallel
        m_HatTangentInf = 0.;
    }
    const double heightSup = sourceOffsetY + mag * (corner2[1] + projOffsetY - sourceOffsetY);
    if (heightSup < m_HatHeightSup)
    {
      m_HatHeightSup = heightSup;
      m_HatTangentSup = m_HatHeightSup / sid;
      if (sdd == 0.) // Parallel
        m_HatTangentSup = 0.;
    }
  }
}

template <class TInputImage, class TOutputImage>
void
FieldOfViewImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  typename TInputImage::DirectionType d = this->GetInput()->GetDirection();
  if (d[0][0] == 1. && d[0][1] == 0. && d[0][2] == 0. && d[1][0] == 0. && d[1][1] == 1. && d[1][2] == 0. &&
      d[2][0] == 0. && d[2][1] == 0. && d[2][2] == 1.)

  {
    // Prepare point increment (TransformIndexToPhysicalPoint too slow)
    typename TInputImage::PointType pointBase, pointIncrement;
    typename TInputImage::IndexType index = outputRegionForThread.GetIndex();
    this->GetInput()->TransformIndexToPhysicalPoint(index, pointBase);
    for (unsigned int i = 0; i < TInputImage::GetImageDimension(); i++)
      index[i]++;
    this->GetInput()->TransformIndexToPhysicalPoint(index, pointIncrement);
    for (unsigned int i = 0; i < TInputImage::GetImageDimension(); i++)
      pointIncrement[i] -= pointBase[i];

    // Iterators
    using InputConstIterator = itk::ImageRegionConstIterator<TInputImage>;
    InputConstIterator itIn(this->GetInput(0), outputRegionForThread);
    itIn.GoToBegin();
    using OutputIterator = itk::ImageRegionIterator<TOutputImage>;
    OutputIterator itOut(this->GetOutput(), outputRegionForThread);
    itOut.GoToBegin();

    // Go over output, compute weights and avoid redundant computation
    typename TInputImage::PointType point = pointBase;
    for (unsigned int k = 0; k < outputRegionForThread.GetSize(2); k++)
    {
      double zsquare = m_CenterZ - point[2];
      zsquare *= zsquare;
      point[1] = pointBase[1];
      for (unsigned int j = 0; j < outputRegionForThread.GetSize(1); j++)
      {
        point[0] = pointBase[0];
        for (unsigned int i = 0; i < outputRegionForThread.GetSize(0); i++)
        {
          double xsquare = m_CenterX - point[0];
          xsquare *= xsquare;
          double radius = std::sqrt(xsquare + zsquare);
          if (radius <= m_Radius && radius * m_HatTangentInf >= m_HatHeightInf - point[1] &&
              radius * m_HatTangentSup <= m_HatHeightSup - point[1])
          {
            if (m_Mask)
              itOut.Set(this->m_InsideValue);
            else
              itOut.Set(itIn.Get());
          }
          else
            itOut.Set(this->m_OutsideValue);
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
    using InputConstIterator = itk::ImageRegionConstIteratorWithIndex<TInputImage>;
    InputConstIterator itIn(this->GetInput(0), outputRegionForThread);

    using OutputIterator = itk::ImageRegionIterator<TOutputImage>;
    OutputIterator itOut(this->GetOutput(), outputRegionForThread);

    typename TInputImage::PointType point;
    while (!itIn.IsAtEnd())
    {
      this->GetInput()->TransformIndexToPhysicalPoint(itIn.GetIndex(), point);
      double radius = std::sqrt(point[0] * point[0] + point[2] * point[2]);
      if (radius <= m_Radius && point[1] <= m_HatHeightSup - radius * m_HatTangentSup &&
          point[1] >= m_HatHeightInf - radius * m_HatTangentInf)
      {
        if (m_Mask)
          itOut.Set(this->m_InsideValue);
        else
          itOut.Set(itIn.Get());
      }
      else
        itOut.Set(this->m_OutsideValue);
      ++itIn;
      ++itOut;
    }
  }
}

template <class TInputImage, class TOutputImage>
void
FieldOfViewImageFilter<TInputImage, TOutputImage>::AddCollimationConstraints(const FOVRadiusType type, _lprec * lp)
{
  constexpr int Ncol = 3;
  int           colno[Ncol] = { 1, 2, 3 };
  REAL          row[Ncol];
  for (unsigned int iProj = 0; iProj < m_Geometry->GetGantryAngles().size(); iProj++)
  {
    const double X1 = m_Geometry->GetCollimationUInf()[iProj];
    const double X2 = m_Geometry->GetCollimationUSup()[iProj];
    if (X1 == std::numeric_limits<double>::max() && X2 == std::numeric_limits<double>::max())
    {
      continue;
    }
    if (X1 == std::numeric_limits<double>::max() || X2 == std::numeric_limits<double>::max())
    {
      itkWarningMacro("Having only one jaw that is not at the default value is unexpected.");
    }

    // Compute 3D position of jaws
    using PointType = typename GeometryType::VectorType;
    typename GeometryType::HomogeneousVectorType sourceH = m_Geometry->GetSourcePosition(iProj);
    PointType                                    source(0.);
    source[0] = sourceH[0];
    source[2] = sourceH[2];
    double    sourceNorm = source.GetNorm();
    PointType sourceDir = source / sourceNorm;

    PointType v(0.);
    v[1] = 1.;
    PointType u = CrossProduct(v, sourceDir);

    // Compute the equation of a line of the ax+by=c
    // https://en.wikipedia.org/wiki/Linear_equation#Two-point_form
    // Then compute the coefficient in front of r as suggested in
    // http://www.ifor.math.ethz.ch/teaching/lectures/intro_ss11/Exercises/solutionEx11-12.pdf
    PointType inf = u * -1. * X1;
    double    aInf = inf[2] - source[2];
    double    bInf = source[0] - inf[0];
    double    cInf = source[0] * inf[2] - inf[0] * source[2];
    double    dInf = std::sqrt(aInf * aInf + bInf * bInf);

    PointType sup = u * X2;
    double    aSup = sup[2] - source[2];
    double    bSup = source[0] - sup[0];
    double    cSup = source[0] * sup[2] - sup[0] * source[2];
    double    dSup = std::sqrt(aSup * aSup + bSup * bSup);

    // Check on corners
    if (aInf * sup[0] + bInf * sup[2] >= cInf)
    {
      aInf *= -1.;
      bInf *= -1.;
      cInf *= -1.;
    }
    else if (aSup * inf[0] + bSup * inf[2] >= cSup)
    {
      aSup *= -1.;
      bSup *= -1.;
      cSup *= -1.;
    }
    else
    {
      itkExceptionMacro(<< "Something's wrong with the jaw handling.");
    }

    // Now add the constraints of the form ax+by+dr<=c
    if (type == RADIUSINF || type == RADIUSBOTH)
    {
      row[0] = aInf;
      row[1] = bInf;
      row[2] = dInf;
      if (!add_constraintex(lp, 3, row, colno, LE, cInf))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
    }
    if (type == RADIUSSUP || type == RADIUSBOTH)
    {
      row[0] = aSup;
      row[1] = bSup;
      row[2] = dSup;
      if (!add_constraintex(lp, 3, row, colno, LE, cSup))
        itkExceptionMacro(<< "Couldn't add simplex constraint");
    }
  }
}

} // end namespace rtk

#endif // rtkFieldOfViewImageFilter_hxx
