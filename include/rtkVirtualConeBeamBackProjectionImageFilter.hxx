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

#ifndef rtkVirtualConeBeamBackProjectionImageFilter_hxx
#define rtkVirtualConeBeamBackProjectionImageFilter_hxx

#include "math.h"

#include "rtkVirtualConeBeamBackProjectionImageFilter.h"

#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>

#define BILINEAR_BACKPROJECTION

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
VirtualConeBeamBackProjectionImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{
  // Check if detector is cylindrical
  if (this->m_Geometry->GetRadiusCylindricalDetector() != 0)
  {
    itkGenericExceptionMacro(
      << "Voxel-based VirtualConeBeam back projector can currently not handle cylindrical detectors");
  }

  // Run superclass' GenerateOutputInformation
  Superclass::GenerateOutputInformation();
}

/**
 * GenerateData performs the accumulation
 */
template <class TInputImage, class TOutputImage>
void
VirtualConeBeamBackProjectionImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension - 1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension - 1);

  // Create interpolator, could be any interpolation
  using InterpolatorType = itk::LinearInterpolateImageFunction<ProjectionImageType, double>;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

  // Iterators on volume input and output
  using InputRegionIterator = itk::ImageRegionConstIterator<TInputImage>;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<TOutputImage>;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Initialize output region with input region in case the filter is not in
  // place
  if (this->GetInput() != this->GetOutput())
  {
    itIn.GoToBegin();
    while (!itIn.IsAtEnd())
    {
      itOut.Set(itIn.Get());
      ++itIn;
      ++itOut;
    }
  }

  // Rotation center (assumed to be at 0 yet)
  typename TInputImage::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);
  itk::ContinuousIndex<double, Dimension> rotCenterIndex;
  this->GetInput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);

  // Continuous index at which we interpolate
  itk::ContinuousIndex<double, Dimension - 1> pointProj;

  itk::Matrix<double, Dimension + 1, Dimension + 1> matrixVol =
    GetIndexToPhysicalPointMatrix<TOutputImage>(this->GetOutput());

  // Go over each projection
  for (unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
  {
    // Extract the current slice
    ProjectionImagePointer projection;
    projection = this->template GetProjection<ProjectionImageType>(iProj);
    interpolator->SetInputImage(projection);

    // Index to index matrix normalized to have a correct distance, i.e., SID at
    // the isocenter
    ProjectionMatrixType matrix = this->GetIndexToIndexProjectionMatrix(iProj);
    double               perspFactor = matrix[Dimension - 1][Dimension];
    for (unsigned int j = 0; j < Dimension; j++)
      perspFactor += matrix[Dimension - 1][j] * rotCenterIndex[j];
    matrix *= this->GetGeometry()->GetSourceToIsocenterDistances()[iProj] / perspFactor;

    RotationMatrixType rotMat(this->m_Geometry->GetSourceTranslationMatrices()[iProj].GetVnlMatrix() *
                              this->m_Geometry->GetRotationMatrices()[iProj].GetVnlMatrix() * matrixVol.GetVnlMatrix());

    // Optimized version
    if (fabs(matrix[1][0]) < 1e-10 && fabs(matrix[2][0]) < 1e-10)
    {
      OptimizedBackprojectionX(outputRegionForThread, matrix, projection, rotMat);
      continue;
    }
    if (fabs(matrix[1][1]) < 1e-10 && fabs(matrix[2][1]) < 1e-10)
    {
      OptimizedBackprojectionY(outputRegionForThread, matrix, projection, rotMat);
      continue;
    }

    // Go over each voxel
    itOut.GoToBegin();
    while (!itOut.IsAtEnd())
    {
      // Compute projection index
      for (unsigned int i = 0; i < Dimension - 1; i++)
      {
        pointProj[i] = matrix[i][Dimension];
        for (unsigned int j = 0; j < Dimension; j++)
          pointProj[i] += matrix[i][j] * itOut.GetIndex()[j];
      }

      // Apply perspective
      double perspFactor_local = matrix[Dimension - 1][Dimension];
      for (unsigned int j = 0; j < Dimension; j++)
        perspFactor_local += matrix[Dimension - 1][j] * itOut.GetIndex()[j];
      double perspFactor_inv = 1 / perspFactor_local;
      for (unsigned int i = 0; i < Dimension - 1; i++)
        pointProj[i] = pointProj[i] * perspFactor_inv;

      // Interpolate if in projection
      if (interpolator->IsInsideBuffer(pointProj))
      {
        const double x = rotMat[0][0] * itOut.GetIndex()[0] + rotMat[0][1] * itOut.GetIndex()[1] +
                         rotMat[0][2] * itOut.GetIndex()[2] + rotMat[0][3];
        const double wgt = 1. / std::sqrt(x * x + perspFactor_local * perspFactor_local);
        itOut.Set(itOut.Get() + wgt * interpolator->EvaluateAtContinuousIndex(pointProj));
      }

      ++itOut;
    }
  }
}

template <class TInputImage, class TOutputImage>
void
VirtualConeBeamBackProjectionImageFilter<TInputImage, TOutputImage>::OptimizedBackprojectionX(
  const OutputImageRegionType & region,
  const ProjectionMatrixType &  matrix,
  const ProjectionImagePointer  projection,
  const RotationMatrixType &    rotMat)
{
  typename ProjectionImageType::SizeType  pSize = projection->GetBufferedRegion().GetSize();
  typename ProjectionImageType::IndexType pIndex = projection->GetBufferedRegion().GetIndex();
  typename TOutputImage::SizeType         vBufferSize = this->GetOutput()->GetBufferedRegion().GetSize();
  typename TOutputImage::IndexType        vBufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
  typename TInputImage::PixelType *       pProj = nullptr;
  typename TOutputImage::PixelType *      pVol = nullptr, *pVolZeroPointer = nullptr;

  // Pointers in memory to index (0,0,0) which do not necessarily exist
  pVolZeroPointer = this->GetOutput()->GetBufferPointer();
  pVolZeroPointer -= vBufferIndex[0] + vBufferSize[0] * (vBufferIndex[1] + vBufferSize[1] * vBufferIndex[2]);

  // Continuous index at which we interpolate
  double u = NAN, v = NAN, w = NAN;
  int    ui = 0, vi = 0;
  double du = NAN;

  for (int k = region.GetIndex(2); k < region.GetIndex(2) + (int)region.GetSize(2); k++)
  {
    for (int j = region.GetIndex(1); j < region.GetIndex(1) + (int)region.GetSize(1); j++)
    {
      int  i = region.GetIndex(0);
      auto x = rotMat[0][0] * i + rotMat[0][1] * j + rotMat[0][2] * k + rotMat[0][3];
      u = matrix[0][0] * i + matrix[0][1] * j + matrix[0][2] * k + matrix[0][3];
      v = matrix[1][1] * j + matrix[1][2] * k + matrix[1][3];
      w = matrix[2][1] * j + matrix[2][2] * k + matrix[2][3];

      // Apply perspective
      auto winv = 1. / w;
      u = u * winv - pIndex[0];
      v = v * winv - pIndex[1];
      du = winv * matrix[0][0];
      auto dx = rotMat[0][0];

      double u1 = NAN, u2 = NAN, v1 = NAN, v2 = NAN;
      vi = itk::Math::floor(v);
      if (vi >= 0 && vi < (int)pSize[1] - 1)
      {
        v1 = v - vi;
        v2 = 1. - v1;

        pProj = projection->GetBufferPointer() + vi * pSize[0];
        pVol = pVolZeroPointer + i + vBufferSize[0] * (j + k * vBufferSize[1]);

        // Innermost loop
        for (; i < (region.GetIndex(0) + (int)region.GetSize(0)); i++, x += dx, u += du, pVol++)
        {
          ui = itk::Math::floor(u);
          if (ui >= 0 && ui < (int)pSize[0] - 1)
          {
            const double wgt = 1. / std::sqrt(x * x + w * w);
            u1 = u - ui;
            u2 = 1. - u1;
            *pVol += wgt * (v2 * (u2 * *(pProj + ui) + u1 * *(pProj + ui + 1)) +
                            v1 * (u2 * *(pProj + ui + pSize[0]) + u1 * *(pProj + ui + pSize[0] + 1)));
          }
        } // i
      }
    } // j
  }   // k
}

template <class TInputImage, class TOutputImage>
void
VirtualConeBeamBackProjectionImageFilter<TInputImage, TOutputImage>::OptimizedBackprojectionY(
  const OutputImageRegionType & region,
  const ProjectionMatrixType &  matrix,
  const ProjectionImagePointer  projection,
  const RotationMatrixType &    rotMat)
{
  typename ProjectionImageType::SizeType  pSize = projection->GetBufferedRegion().GetSize();
  typename ProjectionImageType::IndexType pIndex = projection->GetBufferedRegion().GetIndex();
  typename TOutputImage::SizeType         vBufferSize = this->GetOutput()->GetBufferedRegion().GetSize();
  typename TOutputImage::IndexType        vBufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
  typename TInputImage::PixelType *       pProj = nullptr;
  typename TOutputImage::PixelType *      pVol = nullptr, *pVolZeroPointer = nullptr;

  // Pointers in memory to index (0,0,0) which do not necessarily exist
  pVolZeroPointer = this->GetOutput()->GetBufferPointer();
  pVolZeroPointer -= vBufferIndex[0] + vBufferSize[0] * (vBufferIndex[1] + vBufferSize[1] * vBufferIndex[2]);

  // Continuous index at which we interpolate
  double u = NAN, v = NAN, w = NAN;
  int    ui = 0, vi = 0;
  double du = NAN;

  for (int k = region.GetIndex(2); k < region.GetIndex(2) + (int)region.GetSize(2); k++)
  {
    for (int i = region.GetIndex(0); i < region.GetIndex(0) + (int)region.GetSize(0); i++)
    {
      int  j = region.GetIndex(1);
      auto x = rotMat[0][0] * i + rotMat[0][1] * j + rotMat[0][2] * k + rotMat[0][3];
      u = matrix[0][0] * i + matrix[0][1] * j + matrix[0][2] * k + matrix[0][3];
      v = matrix[1][0] * i + matrix[1][2] * k + matrix[1][3];
      w = matrix[2][0] * i + matrix[2][2] * k + matrix[2][3];

      // Apply perspective
      auto winv = 1. / w;
      u = u * winv - pIndex[0];
      v = v * winv - pIndex[1];
      du = winv * matrix[0][1];
      auto dx = rotMat[0][1];

      vi = itk::Math::floor(v);
      if (vi >= 0 && vi < (int)pSize[1] - 1)
      {
        pVol = pVolZeroPointer + i + vBufferSize[0] * (j + k * vBufferSize[1]);
        for (; j < (region.GetIndex(1) + (int)region.GetSize(1)); j++, pVol += vBufferSize[0], u += du, x += dx)
        {
          ui = itk::Math::floor(u);
          if (ui >= 0 && ui < (int)pSize[0] - 1)
          {
            const double wgt = 1. / std::sqrt(x * x + w * w);
            double       u1 = NAN, u2 = NAN, v1 = NAN, v2 = NAN;
            pProj = projection->GetBufferPointer() + vi * pSize[0] + ui;
            v1 = v - vi;
            v2 = 1. - v1;
            u1 = u - ui;
            u2 = 1. - u1;
            *pVol += wgt * (v2 * (u2 * *(pProj) + u1 * *(pProj + 1)) +
                            v1 * (u2 * *(pProj + pSize[0]) + u1 * *(pProj + pSize[0] + 1)));
          }
        } // j
      }
    } // i
  }   // k
}

} // end namespace rtk

#endif
