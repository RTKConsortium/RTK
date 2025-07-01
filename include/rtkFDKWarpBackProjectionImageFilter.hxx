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

#ifndef rtkFDKWarpBackProjectionImageFilter_hxx
#define rtkFDKWarpBackProjectionImageFilter_hxx


#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>

#define BILINEAR_BACKPROJECTION

namespace rtk
{

template <class TInputImage, class TOutputImage, class TDeformation>
FDKWarpBackProjectionImageFilter<TInputImage, TOutputImage, TDeformation>::FDKWarpBackProjectionImageFilter() = default;

template <class TInputImage, class TOutputImage, class TDeformation>
void
FDKWarpBackProjectionImageFilter<TInputImage, TOutputImage, TDeformation>::GenerateData()
{
  this->AllocateOutputs();
  this->SetTranspose(true);

  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension - 1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension - 1);

  // Initialize output region with input region in case the filter is not in
  // place
  this->GetMultiThreader()->SetNumberOfWorkUnits(this->GetNumberOfWorkUnits());
  if (this->GetInput() != this->GetOutput())
  {

    this->GetMultiThreader()->template ParallelizeImageRegion<TOutputImage::ImageDimension>(
      this->GetOutput()->GetRequestedRegion(),
      [this](const typename TOutputImage::RegionType & outputRegionForThread) {
        // Iterators on volume input and output
        using InputRegionIterator = itk::ImageRegionConstIterator<TInputImage>;
        InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
        using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<TOutputImage>;
        OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

        itIn.GoToBegin();
        while (!itIn.IsAtEnd())
        {
          itOut.Set(itIn.Get());
          ++itIn;
          ++itOut;
        }
      },
      nullptr);
  }

  // Rotation center (assumed to be at 0 yet)
  typename TInputImage::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);

  // Warped point and interpolator for vector field
  using WarpInterpolatorType = itk::LinearInterpolateImageFunction<typename TDeformation::OutputImageType, double>;
  auto                                              warpInterpolator = WarpInterpolatorType::New();
  itk::Matrix<double, Dimension + 1, Dimension + 1> matrixVol =
    GetPhysicalPointToIndexMatrix<TOutputImage>(this->GetOutput());

  // Go over each projection
  for (unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
  {
    // Set the deformation
    m_Deformation->SetFrame(iProj);
    m_Deformation->Update();
    warpInterpolator->SetInputImage(m_Deformation->GetOutput());

    // Extract the current slice and create interpolator, could be any interpolation
    ProjectionImagePointer projection = this->template GetProjection<ProjectionImageType>(iProj);
    using InterpolatorType = itk::LinearInterpolateImageFunction<ProjectionImageType, double>;
    auto interpolator = InterpolatorType::New();
    interpolator->SetInputImage(projection);

    // Index to index matrix normalized to have a correct backprojection weight
    // (1 at the isocenter)
    ProjectionMatrixType matrix(this->GetIndexToIndexProjectionMatrix(iProj).GetVnlMatrix() * matrixVol.GetVnlMatrix());
    double               perspFactor = matrix[Dimension - 1][Dimension];
    for (unsigned int j = 0; j < Dimension; j++)
      perspFactor += matrix[Dimension - 1][j] * rotCenterPoint[j];
    matrix /= perspFactor;

    this->GetMultiThreader()->template ParallelizeImageRegion<TOutputImage::ImageDimension>(
      this->GetOutput()->GetRequestedRegion(),
      [this, warpInterpolator, interpolator, matrix](const typename TOutputImage::RegionType & outputRegionForThread) {
        using OutputRegionIterator = itk::ImageRegionIteratorWithIndex<TOutputImage>;
        OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

        // Go over each voxel
        itOut.GoToBegin();
        while (!itOut.IsAtEnd())
        {
          // Warp
          typename TOutputImage::PointType point;
          this->GetOutput()->TransformIndexToPhysicalPoint(itOut.GetIndex(), point);
          if (warpInterpolator->IsInsideBuffer(point))
            point = point + warpInterpolator->Evaluate(point);

          // Compute projection index
          itk::ContinuousIndex<double, TOutputImage::ImageDimension - 1> pointProj;
          for (unsigned int i = 0; i < TOutputImage::ImageDimension - 1; i++)
          {
            pointProj[i] = matrix[i][TOutputImage::ImageDimension];
            for (unsigned int j = 0; j < TOutputImage::ImageDimension; j++)
              pointProj[i] += matrix[i][j] * point[j];
          }

          // Apply perspective
          double perspFactor_local = matrix[TOutputImage::ImageDimension - 1][TOutputImage::ImageDimension];
          for (unsigned int j = 0; j < TOutputImage::ImageDimension; j++)
            perspFactor_local += matrix[TOutputImage::ImageDimension - 1][j] * point[j];
          perspFactor_local = 1 / perspFactor_local;
          for (unsigned int i = 0; i < TOutputImage::ImageDimension - 1; i++)
            pointProj[i] = pointProj[i] * perspFactor_local;

          // Interpolate if in projection
          if (interpolator->IsInsideBuffer(pointProj))
          {
            itOut.Set(itOut.Get() +
                      perspFactor_local * perspFactor_local * interpolator->EvaluateAtContinuousIndex(pointProj));
          }

          ++itOut;
        }
      },
      nullptr);
  }
}

} // end namespace rtk

#endif
