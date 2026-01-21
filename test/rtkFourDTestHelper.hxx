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
#ifndef rtkFourDTestHelper_hxx
#define rtkFourDTestHelper_hxx

#include <fstream>

#include <itkPasteImageFilter.h>
#include <itkJoinSeriesImageFilter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageDuplicator.h>

#include "rtkConstantImageSource.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkTest.h"

namespace rtk
{

template <class PixelType>
FourDTestData<PixelType>
GenerateFourDTestData(bool fastTests)
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
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 5;
#else
  constexpr unsigned int NumberOfProjectionImages = 64;
#endif

  FourDTestData<PixelType> data;

  /* =========================
   *  Constant volume sources
   * ========================= */
  using ConstantVolumeSourceType = rtk::ConstantImageSource<VolumeType>;
  auto volumeSource = ConstantVolumeSourceType::New();

  auto origin = itk::MakePoint(-63., -31., -63.);
  auto spacing = fastTests ? itk::MakeVector(16., 8., 16.) : itk::MakeVector(4., 4., 4.);
  auto size = fastTests ? itk::MakeSize(8, 8, 8) : itk::MakeSize(32, 16, 32);

  volumeSource->SetOrigin(origin);
  volumeSource->SetSpacing(spacing);
  volumeSource->SetSize(size);
  volumeSource->SetConstant(0.);
  volumeSource->Update();

  data.SingleVolume = volumeSource->GetOutput();

  using ConstantFourDSourceType = rtk::ConstantImageSource<VolumeSeriesType>;
  auto fourDSource = ConstantFourDSourceType::New();

  auto fourDOrigin = itk::MakePoint(-63., -31., -63., 0.);
  auto fourDSpacing = fastTests ? itk::MakeVector(16., 8., 16., 1.) : itk::MakeVector(4., 4., 4., 1.);
  auto fourDSize = fastTests ? itk::MakeSize(8, 8, 8, 2) : itk::MakeSize(32, 16, 32, 8);

  fourDSource->SetOrigin(fourDOrigin);
  fourDSource->SetSpacing(fourDSpacing);
  fourDSource->SetSize(fourDSize);
  fourDSource->SetConstant(0.);
  fourDSource->Update();

  data.InitialVolumeSeries = fourDSource->GetOutput();

  /* =========================
   *  Projection accumulation
   * ========================= */
  using ConstantProjectionSourceType = rtk::ConstantImageSource<ProjectionStackType>;
  auto projectionsSource = ConstantProjectionSourceType::New();

  auto projOrigin = itk::MakePoint(-254., -254., -254.);
  auto projSpacing = fastTests ? itk::MakeVector(32., 32., 32.) : itk::MakeVector(8., 8., 1.);
  auto projSize =
    fastTests ? itk::MakeSize(32, 32, NumberOfProjectionImages) : itk::MakeSize(64, 64, NumberOfProjectionImages);

  projectionsSource->SetOrigin(projOrigin);
  projectionsSource->SetSpacing(projSpacing);
  projectionsSource->SetSize(projSize);
  projectionsSource->SetConstant(0.);
  projectionsSource->Update();

  auto oneProjectionSource = ConstantProjectionSourceType::New();
  projSize[2] = 1;
  oneProjectionSource->SetOrigin(projOrigin);
  oneProjectionSource->SetSpacing(projSpacing);
  oneProjectionSource->SetSize(projSize);
  oneProjectionSource->SetConstant(0.);

  data.Geometry = GeometryType::New();

  using REIType = rtk::RayEllipsoidIntersectionImageFilter<VolumeType, ProjectionStackType>;
  using PasteType = itk::PasteImageFilter<ProjectionStackType, ProjectionStackType, ProjectionStackType>;

  /* Allocate explicit accumulator image */
  auto accumulated = ProjectionStackType::New();
  accumulated->SetOrigin(projectionsSource->GetOutput()->GetOrigin());
  accumulated->SetSpacing(projectionsSource->GetOutput()->GetSpacing());
  accumulated->SetDirection(projectionsSource->GetOutput()->GetDirection());
  accumulated->SetRegions(projectionsSource->GetOutput()->GetLargestPossibleRegion());
  accumulated->Allocate();
  accumulated->FillBuffer(0.);

  auto paste = PasteType::New();
  paste->SetDestinationImage(accumulated);

  itk::Index<3> destIndex;
  destIndex.Fill(0);

  // Create the signal, as a file and as an std::vector
#ifdef USE_CUDA
  data.SignalFileName = "signal_4D_cuda.txt";
#else
  data.SignalFileName = "signal_4D.txt";
#endif

  data.Signal = std::vector<double>();
  std::ofstream signalFile(data.SignalFileName.c_str());

  for (unsigned int i = 0; i < NumberOfProjectionImages; ++i)
  {
    data.Geometry->AddProjection(600., 1200., i * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    auto geom = GeometryType::New();
    geom->AddProjection(600., 1200., i * 360. / NumberOfProjectionImages, 0, 0, 0, 0, 20, 15);

    auto e1 = REIType::New();
    e1->SetInput(oneProjectionSource->GetOutput());
    e1->SetGeometry(geom);
    e1->SetDensity(2.);
    e1->SetAxis(itk::MakeVector(60., 30., 60.));
    e1->SetCenter(itk::MakeVector(0., 0., 0.));
    e1->InPlaceOff();
    e1->Update();

    auto e2 = REIType::New();
    e2->SetInput(e1->GetOutput());
    e2->SetGeometry(geom);
    e2->SetDensity(-1.);
    e2->SetAxis(itk::MakeVector(8., 8., 8.));
    auto center = itk::MakeVector(4 * (itk::Math::abs((4 + i) % 8 - 4.) - 2.), 0., 0.);
    e2->SetCenter(center);
    e2->InPlaceOff();
    e2->Update();

    paste->SetSourceImage(e2->GetOutput());
    paste->SetSourceRegion(e2->GetOutput()->GetLargestPossibleRegion());
    paste->SetDestinationIndex(destIndex);
    paste->Update();

    accumulated = paste->GetOutput();
    paste->SetDestinationImage(accumulated);

    destIndex[2]++;

    data.Signal.push_back((i % 8) / 8.);
    signalFile << data.Signal.back() << std::endl;
  }

  signalFile.close();
  data.Projections = accumulated;

  /* =========================
   *  DVF & inverse DVF
   * ========================= */
  auto dvf = DVFSequenceImageType::New();
  auto idvf = DVFSequenceImageType::New();

  typename DVFSequenceImageType::RegionType region;
  auto                                      dvfSize = itk::MakeSize(fourDSize[0], fourDSize[1], fourDSize[2], 2);
  region.SetSize(dvfSize);

  dvf->SetRegions(region);
  dvf->SetOrigin(fourDOrigin);
  dvf->SetSpacing(fourDSpacing);
  dvf->Allocate();

  idvf->SetRegions(region);
  idvf->SetOrigin(fourDOrigin);
  idvf->SetSpacing(fourDSpacing);
  idvf->Allocate();

  itk::ImageRegionIteratorWithIndex<DVFSequenceImageType> it(dvf, region);
  itk::ImageRegionIteratorWithIndex<DVFSequenceImageType> iit(idvf, region);

  DVFVectorType                            v;
  typename DVFSequenceImageType::IndexType centerIndex;
  centerIndex.Fill(0);
  centerIndex[0] = dvfSize[0] / 2;
  centerIndex[1] = dvfSize[1] / 2;
  centerIndex[2] = dvfSize[2] / 2;

  for (; !it.IsAtEnd(); ++it, ++iit)
  {
    v.Fill(0.);
    auto d = it.GetIndex() - centerIndex;
    if (0.3 * d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 40)
      v[0] = (it.GetIndex()[3] == 0 ? -8. : 8.);
    it.Set(v);
    iit.Set(-v);
  }

  data.DVF = dvf;
  data.InverseDVF = idvf;

  /* =========================
   *  Ground truth
   * ========================= */
  auto join = itk::JoinSeriesImageFilter<VolumeType, VolumeSeriesType>::New();

  for (unsigned int t = 0; t < fourDSize[3]; ++t)
  {
    using DEType = rtk::DrawEllipsoidImageFilter<VolumeType, VolumeType>;

    auto de1 = DEType::New();
    de1->SetInput(volumeSource->GetOutput());
    de1->SetDensity(2.);
    de1->SetAxis(itk::MakeVector(60., 30., 60.));
    de1->SetCenter(itk::MakeVector(0., 0., 0.));
    de1->InPlaceOff();
    de1->Update();

    auto de2 = DEType::New();
    de2->SetInput(de1->GetOutput());
    de2->SetDensity(-1.);
    de2->SetAxis(itk::MakeVector(8., 8., 8.));
    de2->SetCenter(itk::MakeVector(4 * (itk::Math::abs((4 + t) % 8 - 4.) - 2.), 0., 0.));
    de2->InPlaceOff();
    de2->Update();

    using DuplicatorType = itk::ImageDuplicator<VolumeType>;
    auto duplicator = DuplicatorType::New();
    duplicator->SetInputImage(de2->GetOutput());
    duplicator->Update();

    join->SetInput(t, duplicator->GetOutput());
  }

  join->Update();
  data.GroundTruth = join->GetOutput();

  return data;
}

} // namespace rtk

#endif
