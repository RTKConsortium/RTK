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

#ifndef rtkKatsevichBackProjectionImageFilter_hxx
#define rtkKatsevichBackProjectionImageFilter_hxx

#include "math.h"

#include "rtkKatsevichBackProjectionImageFilter.h"

#include <rtkHomogeneousMatrix.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkPixelTraits.h>

namespace rtk
{


template <class TInputImage, class TOutputImage>
KatsevichBackProjectionImageFilter<TInputImage, TOutputImage>::KatsevichBackProjectionImageFilter()
{
  m_Geometry = nullptr;
  m_PILines = nullptr;
  this->SetNumberOfRequiredInputs(2);
  this->SetInPlace(true);
}

template <class TInputImage, class TOutputImage>
void
KatsevichBackProjectionImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
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
    ProjectionMatrixType matrix = this->GetIndexToIndexProjectionMatrix(iProj);

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


  // std::cout << "Corner inf : " << cornerInf << "Corner sup " << cornerSup << std::endl;
  reqRegion.SetIndex(0, itk::Math::floor(cornerInf[0]));
  reqRegion.SetIndex(1, itk::Math::floor(cornerInf[1]));
  reqRegion.SetSize(0, itk::Math::ceil(cornerSup[0] - reqRegion.GetIndex(0)) + 1);
  reqRegion.SetSize(1, itk::Math::ceil(cornerSup[1] - reqRegion.GetIndex(1)) + 1);

  if (reqRegion.Crop(inputPtr1->GetLargestPossibleRegion()))
  {
    inputPtr1->SetRequestedRegion(reqRegion);
  }
  else
  {
    inputPtr1->SetRequestedRegion(inputPtr1->GetLargestPossibleRegion());
  }
}

template <class TInputImage, class TOutputImage>
void
KatsevichBackProjectionImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() ITKv5_CONST
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull() || !this->m_Geometry->GetTheGeometryIsVerified())
    itkExceptionMacro(<< "Geometry has not been set or not been checked");
}


template <class TInputImage, class TOutputImage>
void
KatsevichBackProjectionImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
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

  typename PILineImageFilterType::Pointer pil = PILineImageFilterType::New();
  pil->SetInput(this->GetOutput());
  pil->SetGeometry(this->m_Geometry);
  pil->Update();
  this->m_PILines = pil->GetOutput();
}

/**
 * GenerateData performs the accumulation
 */
template <class TInputImage, class TOutputImage>
void
KatsevichBackProjectionImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
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
  using PILineIterator = itk::ImageRegionConstIterator<PILineImageType>;
  using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<TOutputImage>;

  InputRegionIterator  itIn(this->GetInput(0), outputRegionForThread);
  PILineIterator       itPIL(this->m_PILines, outputRegionForThread);
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

  // Get First Angle Values to map idx_proj -> angles
  double firstAngle = m_Geometry->GetHelicalAngles()[0];
  double delta_lambda = m_Geometry->GetHelixAngularGap();
  double lambda = 0.;

  // Go over each projections
  for (unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
  {
    // Extract the current slice
    ProjectionImagePointer projection = this->template GetProjection<ProjectionImageType>(iProj);

    ProjectionMatrixType matrix = this->GetIndexToIndexProjectionMatrix(iProj);
    interpolator->SetInputImage(projection);

    lambda = firstAngle + iProj * delta_lambda;

    // Cylindrical detector centered on source case
    if (m_Geometry->GetRadiusCylindricalDetector() != 0)
    {
      ProjectionMatrixType volIndexToProjPP = this->GetVolumeIndexToProjectionPhysicalPointMatrix(iProj);
      itk::Matrix<double, TInputImage::ImageDimension, TInputImage::ImageDimension> projPPToProjIndex =
        this->GetProjectionPhysicalPointToProjectionIndexMatrix(iProj);
      CylindricalDetectorCenteredOnSourceBackprojection(
        outputRegionForThread, volIndexToProjPP, projPPToProjIndex, projection, lambda, delta_lambda);
      continue;
    }

    // Optimized version
    if (fabs(matrix[1][0]) < 1e-10 && fabs(matrix[2][0]) < 1e-10)
    {
      OptimizedBackprojectionX(outputRegionForThread, matrix, projection, lambda, delta_lambda);
      continue;
    }
    if (fabs(matrix[1][1]) < 1e-10 && fabs(matrix[2][1]) < 1e-10)
    {
      OptimizedBackprojectionY(outputRegionForThread, matrix, projection, lambda, delta_lambda);
      continue;
    }

    // Go over each voxel
    itOut.GoToBegin();
    itPIL.GoToBegin();
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

        itk::Vector<typename TOutputImage::InternalPixelType, 2> piline = itPIL.Get();
        typename TOutputImage::InternalPixelType                 sb = piline[0];
        typename TOutputImage::InternalPixelType                 st = piline[1];

        double rho = 0;
        double d_in = (lambda - sb) / delta_lambda;
        double d_out = (st - lambda) / delta_lambda;
        // Apply PiLineConditions

        // if (lambda < sb - delta_lambda)
        //  rho = 0;
        // else if (lambda < sb)
        //  rho = 0.5 * (d_in + 1) * (d_in + 1);
        // else if (lambda < sb + delta_lambda)
        //  rho = 0.5 + d_in + 0.5 * d_in * d_in;
        // else if (lambda < st - delta_lambda)
        //  rho = 1;
        // else if (lambda < st)
        //  rho = 0.5 + d_out + 0.5 * d_out * d_out;
        // else if (lambda < st + delta_lambda)
        //  rho = 0.5 * (1 + d_out * (1 + d_out));
        // else
        //  rho = 0;

        if (lambda < sb)
          rho = 0.;
        else if (lambda > sb && lambda < st)
          rho = 1.;
        else
          rho = 0.;

        // Compute normlazation coefficient wstar
        typename TOutputImage::PointType outPoint;
        this->GetOutput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), outPoint);
        double sid = this->GetGeometry()->GetHelixRadius();
        double wstar = sid - outPoint[0] * sin(lambda) - outPoint[2] * cos(lambda);

        itOut.Set(itOut.Get() +
                  rho * delta_lambda * interpolator->EvaluateAtContinuousIndex(pointProj) / (wstar * 2 * M_PI));
      }

      ++itOut;
      ++itPIL;
    }
  }
}


template <class TInputImage, class TOutputImage>
void
KatsevichBackProjectionImageFilter<TInputImage, TOutputImage>::CylindricalDetectorCenteredOnSourceBackprojection(
  const OutputImageRegionType &                                                         region,
  const ProjectionMatrixType &                                                          volIndexToProjPP,
  const itk::Matrix<double, TInputImage::ImageDimension, TInputImage::ImageDimension> & projPPToProjIndex,
  const ProjectionImagePointer                                                          projection,
  const double                                                                          lambda,
  const double                                                                          delta_lambda)
{
  using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<TOutputImage>;
  OutputRegionIterator itOut(this->GetOutput(), region);

  using PILineIterator = itk::ImageRegionConstIterator<PILineImageType>;
  PILineIterator itPIL(this->m_PILines, region);


  const unsigned int Dimension = TInputImage::ImageDimension;

  // Create interpolator, could be any interpolation
  using InterpolatorType = itk::LinearInterpolateImageFunction<ProjectionImageType, double>;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetInputImage(projection);

  // Get radius of the cylindrical detector
  double radius = m_Geometry->GetRadiusCylindricalDetector();

  // Continuous index at which we interpolate
  itk::ContinuousIndex<double, Dimension - 1> pointProj, pointProjIdx;


  // Go over each voxel
  itOut.GoToBegin();
  itPIL.GoToBegin();
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
      itk::Vector<typename TOutputImage::InternalPixelType, 2> piline = itPIL.Get();
      typename TOutputImage::InternalPixelType                 sb = piline[0];
      typename TOutputImage::InternalPixelType                 st = piline[1];

      double rho = 0;
      double d_in = (lambda - sb) / delta_lambda;
      double d_out = (st - lambda) / delta_lambda;
      // Apply PiLineConditions

      // if (lambda < sb - delta_lambda)
      //  rho = 0;
      // else if (lambda < sb)
      //  rho = 0.5 * (d_in + 1) * (d_in + 1);
      // else if (lambda < sb + delta_lambda)
      //  rho = 0.5 + d_in + 0.5 * d_in * d_in;
      // else if (lambda < st - delta_lambda)
      //  rho = 1;
      // else if (lambda < st)
      //  rho = 0.5 + d_out + 0.5 * d_out * d_out;
      // else if (lambda < st + delta_lambda)
      //  rho = 0.5 * (1 + d_out * (1 + d_out));
      // else
      //  rho = 0;
      if (lambda < sb)
        rho = 0.;
      else if (lambda > sb && lambda < st)
        rho = 1.;
      else
        rho = 0.;

      // Compute normlazation coefficient wstar
      typename TOutputImage::PointType outPoint;
      this->GetOutput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), outPoint);
      double sid = this->GetGeometry()->GetHelixRadius();
      double wstar = sid - outPoint[0] * sin(lambda) - outPoint[2] * cos(lambda);

      itOut.Set(itOut.Get() +
                rho * delta_lambda * interpolator->EvaluateAtContinuousIndex(pointProj) / (wstar * 2 * M_PI));
    }

    ++itOut;
    ++itPIL;
  }
}


template <class TInputImage, class TOutputImage>
void
KatsevichBackProjectionImageFilter<TInputImage, TOutputImage>::OptimizedBackprojectionX(
  const OutputImageRegionType & region,
  const ProjectionMatrixType &  matrix,
  const ProjectionImagePointer  projection,
  const double                  lambda,
  const double                  delta_lambda)
{
  typename ProjectionImageType::SizeType    pSize = projection->GetBufferedRegion().GetSize();
  typename ProjectionImageType::IndexType   pIndex = projection->GetBufferedRegion().GetIndex();
  typename TOutputImage::SizeType           vBufferSize = this->GetOutput()->GetBufferedRegion().GetSize();
  typename TOutputImage::IndexType          vBufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
  typename TInputImage::InternalPixelType * pProj = nullptr;
  typename TOutputImage::InternalPixelType *pVol = nullptr, *pVolZeroPointer = nullptr;
  typename TOutputImage::IndexType          indexOutput;
  typename TOutputImage::PointType          outPoint;
  double                                    sid = this->GetGeometry()->GetHelixRadius();


  // Pointers in memory to index (0,0,0) which do not necessarily exist
  pVolZeroPointer = this->GetOutput()->GetBufferPointer();
  pVolZeroPointer -= vBufferIndex[0] + vBufferSize[0] * (vBufferIndex[1] + vBufferSize[1] * vBufferIndex[2]);

  using PILineIterator = itk::ImageRegionConstIterator<PILineImageType>;
  PILineIterator itPIL(this->m_PILines, region);
  itPIL.GoToBegin();


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
            itk::Vector<typename TOutputImage::InternalPixelType, 2> piline = itPIL.Get();
            typename TOutputImage::InternalPixelType                 sb = piline[0];
            typename TOutputImage::InternalPixelType                 st = piline[1];

            double rho = 0;
            double d_in = (lambda - sb) / delta_lambda;
            double d_out = (st - lambda) / delta_lambda;
            // Apply PiLineConditions
            // if (lambda < sb - delta_lambda)
            //  rho = 0;
            // else if (lambda < sb)
            //  rho = 0.5 * (d_in + 1) * (d_in + 1);
            // else if (lambda < sb + delta_lambda)
            //  rho = 0.5 + d_in + 0.5 * d_in * d_in;
            // else if (lambda < st - delta_lambda)
            //  rho = 1;
            // else if (lambda < st)
            //  rho = 0.5 + d_out + 0.5 * d_out * d_out;
            // else if (lambda < st + delta_lambda)
            //  rho = 0.5 * (1 + d_out * (1 + d_out));
            // else
            //  rho = 0;

            if (lambda < sb)
              rho = 0.;
            else if (lambda > sb && lambda < st)
              rho = 1.;
            else
              rho = 0.;


            this->GetOutput()->TransformIndexToPhysicalPoint(indexOutput, outPoint);
            double wstar = sid - outPoint[0] * sin(lambda) - outPoint[2] * cos(lambda);
            u1 = u - ui;
            u2 = 1.0 - u1;
            *pVol += delta_lambda * rho *
                     (v2 * (u2 * *(pProj + ui) + u1 * *(pProj + ui + 1)) +
                      v1 * (u2 * *(pProj + ui + pSize[0]) + u1 * *(pProj + ui + pSize[0] + 1))) /
                     (wstar * 2 * M_PI);
          }
          ++itPIL;
        } // i
      }
    } // j
  }   // k
}

template <class TInputImage, class TOutputImage>
void
KatsevichBackProjectionImageFilter<TInputImage, TOutputImage>::OptimizedBackprojectionY(
  const OutputImageRegionType & region,
  const ProjectionMatrixType &  matrix,
  const ProjectionImagePointer  projection,
  const double                  lambda,
  const double                  delta_lambda)

{
  typename ProjectionImageType::SizeType    pSize = projection->GetBufferedRegion().GetSize();
  typename ProjectionImageType::IndexType   pIndex = projection->GetBufferedRegion().GetIndex();
  typename TOutputImage::SizeType           vBufferSize = this->GetOutput()->GetBufferedRegion().GetSize();
  typename TOutputImage::IndexType          vBufferIndex = this->GetOutput()->GetBufferedRegion().GetIndex();
  typename TInputImage::InternalPixelType * pProj = nullptr;
  typename TOutputImage::InternalPixelType *pVol = nullptr, *pVolZeroPointer = nullptr;
  typename TOutputImage::IndexType          indexOutput;
  typename TOutputImage::PointType          outPoint;
  double                                    sid = this->GetGeometry()->GetHelixRadius();


  // Pointers in memory to index (0,0,0) which do not necessarily exist
  pVolZeroPointer = this->GetOutput()->GetBufferPointer();
  pVolZeroPointer -= vBufferIndex[0] + vBufferSize[0] * (vBufferIndex[1] + vBufferSize[1] * vBufferIndex[2]);

  using PILineIterator = itk::ImageRegionConstIterator<PILineImageType>;
  PILineIterator itPIL(this->m_PILines, region);
  itPIL.GoToBegin();

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
            indexOutput[0] = i;
            indexOutput[1] = j;
            indexOutput[2] = k;
            using ComponentType = typename itk::PixelTraits<typename TInputImage::PixelType>::ValueType;
            ComponentType u1, u2, v1, v2;
            pProj = projection->GetBufferPointer() + vi * pSize[0] + ui;
            v1 = v - vi;
            v2 = 1.0 - v1;
            u1 = u - ui;
            u2 = 1.0 - u1;

            itk::Vector<typename TOutputImage::InternalPixelType, 2> piline = itPIL.Get();
            typename TOutputImage::InternalPixelType                 sb = piline[0];
            typename TOutputImage::InternalPixelType                 st = piline[1];


            double rho = 0;
            double d_in = (lambda - sb) / delta_lambda;
            double d_out = (st - lambda) / delta_lambda;

            // Apply PiLineConditions
            // if (lambda < sb - delta_lambda)
            //  rho = 0;
            // else if (lambda < sb)
            //  rho = 0.5 * (d_in + 1) * (d_in + 1);
            // else if (lambda < sb + delta_lambda)
            //  rho = 0.5 + d_in + 0.5 * d_in * d_in;
            // else if (lambda < st - delta_lambda)
            //  rho = 1;
            // else if (lambda < st)
            //  rho = 0.5 + d_out + 0.5 * d_out * d_out;
            // else if (lambda < st + delta_lambda)
            //  rho = 0.5 * (1 + d_out * (1 + d_out));
            // else
            //  rho = 0;

            if (lambda < sb)
              rho = 0.;
            else if (lambda > sb && lambda < st)
              rho = 1.;
            else
              rho = 0.;

            this->GetOutput()->TransformIndexToPhysicalPoint(indexOutput, outPoint);
            double wstar = sid - outPoint[0] * sin(lambda) - outPoint[2] * cos(lambda);


            *pVol += rho * delta_lambda *
                     (v2 * (u2 * *(pProj) + u1 * *(pProj + 1)) +
                      v1 * (u2 * *(pProj + pSize[0]) + u1 * *(pProj + pSize[0] + 1))) /
                     (wstar * 2 * M_PI);
          }
          ++itPIL;
        } // j
      }
    } // i
  }   // k
}


} // end namespace rtk

#endif
