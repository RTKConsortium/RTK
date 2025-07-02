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

#ifndef rtkBackProjectionImageFilter_hxx
#define rtkBackProjectionImageFilter_hxx

#include "math.h"


#include "rtkHomogeneousMatrix.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkPixelTraits.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  // Input 0 is the volume in which we backproject
  typename Superclass::InputImagePointer inputPtr0 = const_cast<TInputImage *>(this->GetInput(0));
  if (!inputPtr0)
    return;
  inputPtr0->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

  // Input 1 is the stack of projections to backproject
  typename Superclass::InputImagePointer inputPtr1 = const_cast<TInputImage *>(this->GetInput(1));
  if (!inputPtr1)
    return;

  // Geometry size check
  const unsigned int Dimension = TInputImage::ImageDimension;
  const int          lastProjIndex = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension - 1) +
                            this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension - 1);
  if ((int)this->m_Geometry->GetMatrices().size() < lastProjIndex)
  {
    itkExceptionMacro(<< "Mismatch between the number of projections and the geometry entries. "
                      << "Geometry has " << this->m_Geometry->GetMatrices().size()
                      << " entries, which is less than the "
                      << "last index of the projections stack, i.e., " << lastProjIndex << ".");
  }

  typename TInputImage::RegionType reqRegion = inputPtr1->GetLargestPossibleRegion();
  if (m_Geometry.GetPointer() == nullptr || m_Geometry->GetRadiusCylindricalDetector() != 0)
  {
    inputPtr1->SetRequestedRegion(inputPtr1->GetLargestPossibleRegion());
    return;
  }

  itk::ContinuousIndex<double, Dimension> cornerInf;
  itk::ContinuousIndex<double, Dimension> cornerSup;
  cornerInf[0] = itk::NumericTraits<double>::max();
  cornerSup[0] = itk::NumericTraits<double>::NonpositiveMin();
  cornerInf[1] = itk::NumericTraits<double>::max();
  cornerSup[1] = itk::NumericTraits<double>::NonpositiveMin();
  cornerInf[2] = reqRegion.GetIndex(2);
  cornerSup[2] = reqRegion.GetIndex(2) + reqRegion.GetSize(2);

  // Go over each projection
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension - 1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension - 1);
  this->SetTranspose(false);
  for (unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
  {
    // Extract the current slice
    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj);

    // Check which part of the projection image will be backprojected in the
    // volume.
    double firstPerspFactor = 0.;
    int    c[Dimension] = { 0 };
    for (c[2] = 0; c[2] < 2; c[2]++)
      for (c[1] = 0; c[1] < 2; c[1]++)
        for (c[0] = 0; c[0] < 2; c[0]++)
        {
          // Compute corner index
          const double                            eps = 1e-4;
          itk::ContinuousIndex<double, Dimension> index;
          for (unsigned int i = 0; i < Dimension; i++)
          {
            index[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i] - eps;
            index[i] += c[i] * (this->GetOutput()->GetRequestedRegion().GetSize(i) - 1 + 2 * eps);
          }

          // Compute projection index
          itk::ContinuousIndex<double, Dimension - 1> point;
          for (unsigned int i = 0; i < Dimension - 1; i++)
          {
            point[i] = matrix[i][Dimension];
            for (unsigned int j = 0; j < Dimension; j++)
              point[i] += matrix[i][j] * index[j];
          }

          // Apply perspective
          double perspFactor = matrix[Dimension - 1][Dimension];
          for (unsigned int j = 0; j < Dimension; j++)
            perspFactor += matrix[Dimension - 1][j] * index[j];
          perspFactor = 1 / perspFactor;
          for (unsigned int i = 0; i < Dimension - 1; i++)
            point[i] = point[i] * perspFactor;

          // Check if corners all have the same perspective factor sign.
          // If not, source is too close for easily computing a smaller requested
          // region than the largest possible one.
          if (c[0] + c[1] + c[2] == 0)
            firstPerspFactor = perspFactor;
          if (perspFactor * firstPerspFactor < 0.) // Change of sign
          {
            inputPtr1->SetRequestedRegion(inputPtr1->GetLargestPossibleRegion());
            return;
          }

          // Look for extremas on projection to calculate requested region
          for (int i = 0; i < 2; i++)
          {
            cornerInf[i] = std::min(cornerInf[i], point[i]);
            cornerSup[i] = std::max(cornerSup[i], point[i]);
          }
        }
  }
  reqRegion.SetIndex(0, itk::Math::floor(cornerInf[0]));
  reqRegion.SetIndex(1, itk::Math::floor(cornerInf[1]));
  reqRegion.SetSize(0, itk::Math::ceil(cornerSup[0] - reqRegion.GetIndex(0)) + 1);
  reqRegion.SetSize(1, itk::Math::ceil(cornerSup[1] - reqRegion.GetIndex(1)) + 1);

  if (reqRegion.Crop(inputPtr1->GetLargestPossibleRegion()))
    inputPtr1->SetRequestedRegion(reqRegion);
  else
  {
    inputPtr1->SetRequestedRegion(inputPtr1->GetLargestPossibleRegion());
  }
}

template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  this->SetTranspose(true);

  // Check if detector is cylindrical
  double radius = m_Geometry->GetRadiusCylindricalDetector();
  if ((radius != 0) && (radius != this->m_Geometry->GetSourceToDetectorDistances()[0]))
  {
    itkGenericExceptionMacro(<< "Voxel-based back projector can currently handle a cylindrical detector only when it "
                                "is centered on the source. "
                             << "Detector radius is " << radius << ", should be "
                             << this->m_Geometry->GetSourceToDetectorDistances()[0]);
  }
}

/**
 * GenerateData performs the accumulation
 */
template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension - 1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension - 1);

  // Create interpolator, could be any interpolation
  using InterpolatorType = itk::LinearInterpolateImageFunction<ProjectionImageType, double>;
  auto interpolator = InterpolatorType::New();

  // Iterators on volume input and output
  itk::ImageRegionConstIterator<TInputImage> itIn(this->GetInput(), outputRegionForThread);
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

  // Continuous index at which we interpolate
  itk::ContinuousIndex<double, Dimension - 1> pointProj;

  // Go over each projection
  for (unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
  {
    // Extract the current slice
    ProjectionImagePointer projection = GetProjection<ProjectionImageType>(iProj);

    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj);
    interpolator->SetInputImage(projection);

    // Cylindrical detector centered on source case
    if (m_Geometry->GetRadiusCylindricalDetector() != 0)
    {
      ProjectionMatrixType volIndexToProjPP = GetVolumeIndexToProjectionPhysicalPointMatrix(iProj);
      itk::Matrix<double, TInputImage::ImageDimension, TInputImage::ImageDimension> projPPToProjIndex =
        GetProjectionPhysicalPointToProjectionIndexMatrix(iProj);
      CylindricalDetectorCenteredOnSourceBackprojection(
        outputRegionForThread, volIndexToProjPP, projPPToProjIndex, projection);
      continue;
    }

    // Optimized version
    if (itk::Math::abs(matrix[1][0]) < 1e-10 && itk::Math::abs(matrix[2][0]) < 1e-10)
    {
      OptimizedBackprojectionX(outputRegionForThread, matrix, projection);
      continue;
    }
    if (itk::Math::abs(matrix[1][1]) < 1e-10 && itk::Math::abs(matrix[2][1]) < 1e-10)
    {
      OptimizedBackprojectionY(outputRegionForThread, matrix, projection);
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
      double perspFactor = matrix[Dimension - 1][Dimension];
      for (unsigned int j = 0; j < Dimension; j++)
        perspFactor += matrix[Dimension - 1][j] * itOut.GetIndex()[j];
      perspFactor = 1 / perspFactor;
      for (unsigned int i = 0; i < Dimension - 1; i++)
        pointProj[i] = pointProj[i] * perspFactor;

      // Interpolate if in projection
      if (interpolator->IsInsideBuffer(pointProj))
      {
        typename TOutputImage::PixelType v = interpolator->EvaluateAtContinuousIndex(pointProj);
        itOut.Set(itOut.Get() + v);
      }

      ++itOut;
    }
  }
}

template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage, TOutputImage>::CylindricalDetectorCenteredOnSourceBackprojection(
  const OutputImageRegionType &                                                         region,
  const ProjectionMatrixType &                                                          volIndexToProjPP,
  const itk::Matrix<double, TInputImage::ImageDimension, TInputImage::ImageDimension> & projPPToProjIndex,
  const ProjectionImagePointer                                                          projection)
{
  using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<TOutputImage>;
  OutputRegionIterator itOut(this->GetOutput(), region);

  const unsigned int Dimension = TInputImage::ImageDimension;

  // Create interpolator, could be any interpolation
  using InterpolatorType = itk::LinearInterpolateImageFunction<ProjectionImageType, double>;
  auto interpolator = InterpolatorType::New();
  interpolator->SetInputImage(projection);

  // Get radius of the cylindrical detector
  double radius = m_Geometry->GetRadiusCylindricalDetector();

  // Continuous index at which we interpolate
  itk::ContinuousIndex<double, Dimension - 1> pointProj, pointProjIdx;

  // Go over each voxel
  itOut.GoToBegin();
  while (!itOut.IsAtEnd())
  {
    // Compute projection index
    for (unsigned int i = 0; i < Dimension - 1; i++)
    {
      pointProj[i] = volIndexToProjPP[i][Dimension];
      for (unsigned int j = 0; j < Dimension; j++)
        pointProj[i] += volIndexToProjPP[i][j] * itOut.GetIndex()[j];
    }

    // Apply perspective
    double perspFactor = volIndexToProjPP[Dimension - 1][Dimension];
    for (unsigned int j = 0; j < Dimension; j++)
      perspFactor += volIndexToProjPP[Dimension - 1][j] * itOut.GetIndex()[j];
    perspFactor = 1 / perspFactor;
    for (unsigned int i = 0; i < Dimension - 1; i++)
      pointProj[i] = pointProj[i] * perspFactor;

    // Apply correction for cylindrical centered on source
    const double u = pointProj[0];
    pointProj[0] = radius * atan2(u, radius);
    pointProj[1] = pointProj[1] * radius / sqrt(radius * radius + u * u);

    // Convert to projection index
    for (unsigned int i = 0; i < Dimension - 1; i++)
    {
      pointProjIdx[i] = projPPToProjIndex[i][Dimension - 1];
      for (unsigned int j = 0; j < Dimension - 1; j++)
        pointProjIdx[i] += projPPToProjIndex[i][j] * pointProj[j];
    }

    // Interpolate if in projection
    if (interpolator->IsInsideBuffer(pointProjIdx))
    {
      typename TOutputImage::PixelType v = interpolator->EvaluateAtContinuousIndex(pointProjIdx);
      itOut.Set(itOut.Get() + v);
    }

    ++itOut;
  }
}


template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage, TOutputImage>::OptimizedBackprojectionX(const OutputImageRegionType & region,
                                                                               const ProjectionMatrixType &  matrix,
                                                                               const ProjectionImagePointer  projection)
{
  typename ProjectionImageType::SizeType    pSize = projection->GetBufferedRegion().GetSize();
  typename ProjectionImageType::IndexType   pIndex = projection->GetBufferedRegion().GetIndex();
  typename TOutputImage::SizeType           vBufferSize = this->GetOutput()->GetBufferedRegion().GetSize();
  typename TOutputImage::IndexType          vBufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
  typename TInputImage::InternalPixelType * pProj = nullptr;
  typename TOutputImage::InternalPixelType *pVol = nullptr, *pVolZeroPointer = nullptr;

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
      int i = region.GetIndex(0);
      u = matrix[0][0] * i + matrix[0][1] * j + matrix[0][2] * k + matrix[0][3];
      v = matrix[1][1] * j + matrix[1][2] * k + matrix[1][3];
      w = matrix[2][1] * j + matrix[2][2] * k + matrix[2][3];

      // Apply perspective
      w = 1 / w;
      u = u * w - pIndex[0];
      v = v * w - pIndex[1];
      du = w * matrix[0][0];

      using ComponentType = typename itk::PixelTraits<typename TInputImage::PixelType>::ValueType;
      ComponentType u1, u2, v1, v2;
      vi = itk::Math::floor(v);
      if (vi >= 0 && vi < (int)pSize[1] - 1)
      {
        v1 = v - vi;
        v2 = 1.0 - v1;

        pProj = projection->GetBufferPointer() + vi * pSize[0];
        pVol = pVolZeroPointer + i + vBufferSize[0] * (j + k * vBufferSize[1]);

        // Innermost loop
        for (; i < (region.GetIndex(0) + (int)region.GetSize(0)); i++, u += du, pVol++)
        {
          ui = itk::Math::floor(u);
          if (ui >= 0 && ui < (int)pSize[0] - 1)
          {
            u1 = u - ui;
            u2 = 1.0 - u1;
            *pVol += v2 * (u2 * *(pProj + ui) + u1 * *(pProj + ui + 1)) +
                     v1 * (u2 * *(pProj + ui + pSize[0]) + u1 * *(pProj + ui + pSize[0] + 1));
          }
        } // i
      }
    } // j
  } // k
}

template <class TInputImage, class TOutputImage>
void
BackProjectionImageFilter<TInputImage, TOutputImage>::OptimizedBackprojectionY(const OutputImageRegionType & region,
                                                                               const ProjectionMatrixType &  matrix,
                                                                               const ProjectionImagePointer  projection)
{
  typename ProjectionImageType::SizeType    pSize = projection->GetBufferedRegion().GetSize();
  typename ProjectionImageType::IndexType   pIndex = projection->GetBufferedRegion().GetIndex();
  typename TOutputImage::SizeType           vBufferSize = this->GetOutput()->GetBufferedRegion().GetSize();
  typename TOutputImage::IndexType          vBufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
  typename TInputImage::InternalPixelType * pProj = nullptr;
  typename TOutputImage::InternalPixelType *pVol = nullptr, *pVolZeroPointer = nullptr;

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
      int j = region.GetIndex(1);
      u = matrix[0][0] * i + matrix[0][1] * j + matrix[0][2] * k + matrix[0][3];
      v = matrix[1][0] * i + matrix[1][2] * k + matrix[1][3];
      w = matrix[2][0] * i + matrix[2][2] * k + matrix[2][3];

      // Apply perspective
      w = 1 / w;
      u = u * w - pIndex[0];
      v = v * w - pIndex[1];
      du = w * matrix[0][1];

      vi = itk::Math::floor(v);
      if (vi >= 0 && vi < (int)pSize[1] - 1)
      {
        pVol = pVolZeroPointer + i + vBufferSize[0] * (j + k * vBufferSize[1]);
        for (; j < (region.GetIndex(1) + (int)region.GetSize(1)); j++, pVol += vBufferSize[0], u += du)
        {
          ui = itk::Math::floor(u);
          if (ui >= 0 && ui < (int)pSize[0] - 1)
          {
            using ComponentType = typename itk::PixelTraits<typename TInputImage::PixelType>::ValueType;
            ComponentType u1, u2, v1, v2;
            pProj = projection->GetBufferPointer() + vi * pSize[0] + ui;
            v1 = v - vi;
            v2 = 1.0 - v1;
            u1 = u - ui;
            u2 = 1.0 - u1;
            *pVol +=
              v2 * (u2 * *(pProj) + u1 * *(pProj + 1)) + v1 * (u2 * *(pProj + pSize[0]) + u1 * *(pProj + pSize[0] + 1));
          }
        } // j
      }
    } // i
  } // k
}

template <class TInputImage, class TOutputImage>
template <class TProjectionImage>
typename TProjectionImage::Pointer
BackProjectionImageFilter<TInputImage, TOutputImage>::GetProjection(const unsigned int iProj)
{

  typename Superclass::InputImagePointer stack = const_cast<TInputImage *>(this->GetInput(1));

  const int iProjBuff = stack->GetBufferedRegion().GetIndex(ProjectionImageType::ImageDimension);

  auto                                   projection = TProjectionImage::New();
  typename TProjectionImage::RegionType  region;
  typename TProjectionImage::SpacingType spacing;
  typename TProjectionImage::PointType   origin;

  for (unsigned int i = 0; i < TProjectionImage::ImageDimension; i++)
  {
    origin[i] = stack->GetOrigin()[i];
    spacing[i] = stack->GetSpacing()[i];
    region.SetSize(i, stack->GetBufferedRegion().GetSize()[i]);
    region.SetIndex(i, stack->GetBufferedRegion().GetIndex()[i]);
  }
  if (this->GetTranspose())
  {
    typename TProjectionImage::SizeType  size = region.GetSize();
    typename TProjectionImage::IndexType index = region.GetIndex();
    std::swap(size[0], size[1]);
    std::swap(index[0], index[1]);
    std::swap(origin[0], origin[1]);
    std::swap(spacing[0], spacing[1]);
    region.SetSize(size);
    region.SetIndex(index);
  }
  projection->SetSpacing(spacing);
  projection->SetOrigin(origin);
  projection->SetRegions(region);
  projection->Allocate();

  const unsigned int             npixels = projection->GetBufferedRegion().GetNumberOfPixels();
  const InternalInputPixelType * pi = stack->GetBufferPointer() + (iProj - iProjBuff) * npixels;
  InternalInputPixelType *       po = projection->GetBufferPointer();

  // Transpose projection for optimization
  if (this->GetTranspose())
  {
    for (unsigned int j = 0; j < region.GetSize(0); j++, po -= npixels - 1)
      for (unsigned int i = 0; i < region.GetSize(1); i++, po += region.GetSize(0))
        *po = *pi++;
  }
  else
    for (unsigned int i = 0; i < npixels; i++)
      *po++ = *pi++;

  return projection;
}

template <class TInputImage, class TOutputImage>
typename BackProjectionImageFilter<TInputImage, TOutputImage>::ProjectionMatrixType
BackProjectionImageFilter<TInputImage, TOutputImage>::GetIndexToIndexProjectionMatrix(const unsigned int iProj)
{
  const unsigned int Dimension = TInputImage::ImageDimension;

  ProjectionMatrixType VolumeIndexToProjectionPhysicalPointMatrix =
    GetVolumeIndexToProjectionPhysicalPointMatrix(iProj);

  itk::Matrix<double, Dimension, Dimension> ProjectionPhysicalPointToProjectionIndexMatrix =
    GetProjectionPhysicalPointToProjectionIndexMatrix(iProj);

  return ProjectionMatrixType(ProjectionPhysicalPointToProjectionIndexMatrix.GetVnlMatrix() *
                              VolumeIndexToProjectionPhysicalPointMatrix.GetVnlMatrix());
}

template <class TInputImage, class TOutputImage>
typename BackProjectionImageFilter<TInputImage, TOutputImage>::ProjectionMatrixType
BackProjectionImageFilter<TInputImage, TOutputImage>::GetVolumeIndexToProjectionPhysicalPointMatrix(
  const unsigned int iProj)
{
  const unsigned int Dimension = TInputImage::ImageDimension;

  itk::Matrix<double, Dimension + 1, Dimension + 1> matrixVol =
    GetIndexToPhysicalPointMatrix<TOutputImage>(this->GetOutput());

  return ProjectionMatrixType(this->m_Geometry->GetMagnificationMatrices()[iProj].GetVnlMatrix() *
                              this->m_Geometry->GetSourceTranslationMatrices()[iProj].GetVnlMatrix() *
                              this->m_Geometry->GetRotationMatrices()[iProj].GetVnlMatrix() * matrixVol.GetVnlMatrix());
}

template <class TInputImage, class TOutputImage>
itk::Matrix<double, TInputImage::ImageDimension, TInputImage::ImageDimension>
BackProjectionImageFilter<TInputImage, TOutputImage>::GetProjectionPhysicalPointToProjectionIndexMatrix(
  const unsigned int iProj)
{
  const unsigned int Dimension = TInputImage::ImageDimension;

  itk::Matrix<double, Dimension + 1, Dimension + 1> matrixStackProj =
    GetPhysicalPointToIndexMatrix<TOutputImage>(this->GetInput(1));

  itk::Matrix<double, Dimension, Dimension> matrixProj;
  matrixProj.SetIdentity();
  for (unsigned int i = 0; i < Dimension - 1; i++)
  {
    matrixProj[i][Dimension - 1] = matrixStackProj[i][Dimension];
    for (unsigned int j = 0; j < Dimension - 1; j++)
      matrixProj[i][j] = matrixStackProj[i][j];
  }

  // Transpose projection for optimization
  itk::Matrix<double, Dimension, Dimension> matrixFlip;
  matrixFlip.SetIdentity();
  if (this->GetTranspose())
  {
    std::swap(matrixFlip[0][0], matrixFlip[0][1]);
    std::swap(matrixFlip[1][0], matrixFlip[1][1]);
  }

  return itk::Matrix<double, TInputImage::ImageDimension, TInputImage::ImageDimension>(
    matrixFlip.GetVnlMatrix() * matrixProj.GetVnlMatrix() *
    this->m_Geometry->GetProjectionTranslationMatrices()[iProj].GetVnlMatrix());
}


} // end namespace rtk

#endif
