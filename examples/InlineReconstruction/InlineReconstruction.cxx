#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <iomanip>
#include <sstream>
#include <atomic>

#include <itkImageFileWriter.h>
#include <itkExtractImageFilter.h>

#include <rtkFDKConeBeamReconstructionFilter.h>
#include <rtkParkerShortScanImageFilter.h>
#include <rtkProjectionsReader.h>
#include <rtkSheppLoganPhantomFilter.h>
#include <rtkConstantImageSource.h>

// CUDA headers (if available)
#ifdef RTK_USE_CUDA
#  include <rtkCudaFDKConeBeamReconstructionFilter.h>
#  include <rtkCudaParkerShortScanImageFilter.h>
#  include <itkCudaImage.h>
#endif

// Type definitions
using ImageType = itk::Image<float, 3>;
using CudaImageType = itk::CudaImage<float, 3>;

// Constants for acquisition and reconstruction
const int                       numProjections = 64;
const double                    sid = 1000.0;
const double                    sdd = 1500.0;
const double                    arc = 200.0;
const double                    spacing = 1.0;
const int                       imageSize = 256;
const double                    origin = -0.5 * (imageSize - 1);
const std::chrono::milliseconds acquisitionSleepDuration(100);

std::atomic<int> AcquiredProjectionsCount(0);

// Function for the thread simulating an acquisition
// Simulates and writes a projection every 100 ms.
void
Acquisition()
{
  for (int i = 0; i < numProjections; ++i)
  {
    // Create geometry for the acquisition
    auto geometryAcq = rtk::ThreeDCircularProjectionGeometry::New();
    geometryAcq->AddProjection(sid, sdd, i * arc / numProjections);

    // Create a constant image source for the projection
    auto projection = rtk::ConstantImageSource<ImageType>::New();
    projection->SetOrigin(itk::MakePoint(origin, origin, 0.0));
    projection->SetSize(itk::MakeSize(imageSize, imageSize, 1));
    projection->SetSpacing(itk::MakeSpacing(spacing, spacing, spacing));
    projection->Update();

    // Create Shepp-Logan phantom filter
    auto phantom = rtk::SheppLoganPhantomFilter<ImageType, ImageType>::New();
    phantom->SetInput(projection->GetOutput());
    phantom->SetGeometry(geometryAcq);
    phantom->SetPhantomScale(100);

    // Create filename for the output projection
    std::ostringstream filename;
    filename << "projection_" << std::setw(3) << std::setfill('0') << i << ".mha";
    phantom->Update();

    // Write the projection image to file
    itk::WriteImage(phantom->GetOutput(), filename.str());
    AcquiredProjectionsCount++;
    std::this_thread::sleep_for(acquisitionSleepDuration);
  }
}

int
main()
{
  // Launch the simulated acquisition
  std::thread acquisitionThread(Acquisition);

  // Create the geometry for reconstruction
  auto geometryRec = rtk::ThreeDCircularProjectionGeometry::New();
  for (int i = 0; i < numProjections; ++i)
  {
    geometryRec->AddProjection(sid, sdd, i * arc / numProjections);
  }

  // Create the reconstruction pipeline
  auto reader = rtk::ProjectionsReader<ImageType>::New();
  auto extractor = itk::ExtractImageFilter<ImageType, ImageType>::New();
  extractor->SetInput(reader->GetOutput());

  double originValue = origin * sid / sdd;
  double spacingValue = spacing * sid / sdd;

#ifdef RTK_USE_CUDA
  // Use CUDA for Parker short scan image filter
  auto parker = rtk::CudaParkerShortScanImageFilter::New();
  parker->SetGeometry(geometryRec);

  auto reconstructionSource = rtk::ConstantImageSource<CudaImageType>::New();
  reconstructionSource->SetOrigin(itk::MakePoint(originValue, originValue, originValue));
  reconstructionSource->SetSpacing(itk::MakeVector(spacingValue, spacingValue, spacingValue));
  reconstructionSource->SetSize(itk::MakeSize(imageSize, imageSize, imageSize));
  auto fdk = rtk::CudaFDKConeBeamReconstructionFilter::New();
#else
  // Use CPU for Parker short scan image filter
  auto parker = rtk::ParkerShortScanImageFilter<ImageType>::New();
  parker->SetInput(extractor->GetOutput());
  parker->SetGeometry(geometryRec);

  auto reconstructionSource = rtk::ConstantImageSource<ImageType>::New();
  reconstructionSource->SetOrigin(itk::MakePoint(originValue, originValue, originValue));
  reconstructionSource->SetSpacing(itk::MakeVector(spacingValue, spacingValue, spacingValue));
  reconstructionSource->SetSize(itk::MakeSize(imageSize, imageSize, imageSize));
  auto fdk = rtk::FDKConeBeamReconstructionFilter<ImageType>::New();
#endif

  fdk->SetGeometry(geometryRec);
  fdk->SetInput(0, reconstructionSource->GetOutput());
  fdk->SetInput(1, parker->GetOutput());

  // Online reconstruction loop
  int                      reconstructedProjectionsCount = 0;
  std::vector<std::string> projectionFileNames(numProjections, "projection_000.mha");

  while (reconstructedProjectionsCount != numProjections)
  {
    if (reconstructedProjectionsCount < AcquiredProjectionsCount)
    {
      std::cout << "Processing projection #" << reconstructedProjectionsCount << "\r";

      if (reconstructedProjectionsCount == 0)
      {
        reader->SetFileNames(projectionFileNames);
        reader->UpdateOutputInformation();
      }
      else
      {
        // Create filename for the next projection
        std::ostringstream filename;
        filename << "projection_" << std::setw(3) << std::setfill('0') << reconstructedProjectionsCount << ".mha";
        projectionFileNames[reconstructedProjectionsCount] = filename.str();
        reader->SetFileNames(projectionFileNames);

        // Disconnect the pipeline to allow for proper reconstruction
        ImageType::Pointer reconstructedImage = fdk->GetOutput();
        reconstructedImage->DisconnectPipeline();

        fdk->SetInput(0, reconstructionSource->GetOutput());
      }

      // Extract and read the new projection
      auto extractedRegion = reader->GetOutput()->GetLargestPossibleRegion();
      extractedRegion.SetIndex(2, reconstructedProjectionsCount);
      extractedRegion.SetSize(2, 1);
      extractor->SetExtractionRegion(extractedRegion);
      extractor->UpdateLargestPossibleRegion();

#ifdef RTK_USE_CUDA
      // Prepare projection for CUDA processing
      auto projection = CudaImageType::New();
      projection->SetPixelContainer(extractor->GetOutput()->GetPixelContainer());
      projection->CopyInformation(extractor->GetOutput());
      projection->SetBufferedRegion(extractor->GetOutput()->GetBufferedRegion());
      projection->SetRequestedRegion(extractor->GetOutput()->GetRequestedRegion());
      parker->SetInput(projection);
#endif

      fdk->Update();
      reconstructedProjectionsCount++;
    }
    else
    {
      // Sleep briefly to avoid busy waiting
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  acquisitionThread.join();
  itk::WriteImage(fdk->GetOutput(), "fdk.mha");

  return 0;
}
