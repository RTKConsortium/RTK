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
#ifndef rtkFourDTestHelper_h
#define rtkFourDTestHelper_h

#include <string>

#include <itkImage.h>
#include <itkCovariantVector.h>

#ifdef RTK_USE_CUDA
#  include <itkCudaImage.h>
#endif

#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** Structure containing all the test data required for 3D+time filters */
template <class PixelType>
struct FourDTestData
{
  using DVFVectorType = itk::CovariantVector<PixelType, 3>;

#ifdef RTK_USE_CUDA
  using VolumeSeriesType = itk::CudaImage<PixelType, 4>;
  using ProjectionStackType = itk::CudaImage<PixelType, 3>;
  using VolumeType = itk::CudaImage<PixelType, 3>;
  using DVFSequenceImageType = itk::CudaImage<DVFVectorType, 4>;
#else
  using VolumeSeriesType = itk::Image<PixelType, 4>;
  using ProjectionStackType = itk::Image<PixelType, 3>;
  using VolumeType = itk::Image<PixelType, 3>;
  using DVFSequenceImageType = itk::Image<DVFVectorType, 4>;
#endif

  typename VolumeType::Pointer                            SingleVolume;
  typename VolumeSeriesType::Pointer                      InitialVolumeSeries;
  typename ProjectionStackType::Pointer                   Projections;
  typename rtk::ThreeDCircularProjectionGeometry::Pointer Geometry;

  typename DVFSequenceImageType::Pointer DVF;
  typename DVFSequenceImageType::Pointer InverseDVF;

  typename VolumeSeriesType::Pointer GroundTruth;

  std::string         SignalFileName;
  std::vector<double> Signal;
};

/** Generates a full 3D+time dataset (phantom, projections, signal, DVF, ground truth) */
template <class PixelType>
FourDTestData<PixelType>
GenerateFourDTestData(bool fastTests);

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFourDTestHelper.hxx"
#endif

#endif
