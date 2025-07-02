#include "rtkTest.h"

#include "rtkFFTHilbertImageFilter.h"
/**
 * \file rtkhilbertfiltertest.cxx
 *
 * \brief Functional test for the FFT Hilbert filter.
 *
 * This test computes the FFT Hilbert transform of the signal t -> sin(20*pi*t).
 * The computed Hilbert transform is compared against the known analytic Hilbert transform of the signal.
 *
 * \author Aurélien Coussat
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
#ifdef USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

  OutputImageType::SizeType size;
  size[0] = 500;
  size[1] = size[2] = 1;

  OutputImageType::SpacingType spacing;
  spacing[0] = spacing[1] = spacing[2] = .001;

  OutputImageType::IndexType ix;

  // Build the signal t -> sin(20*pi*t)
  auto signal = OutputImageType::New();
  signal->SetSpacing(spacing);
  signal->SetRegions(size);
  signal->Allocate();

  itk::ImageRegionIterator<OutputImageType> signalIt(signal, signal->GetLargestPossibleRegion());

  while (!signalIt.IsAtEnd())
  {
    ix = signalIt.GetIndex();
    double t = ix[0] * spacing[0];
    double v = sin(20 * itk::Math::pi * t);
    signalIt.Set(v);
    ++signalIt;
  }

  // Build the analytic Hilbert transform of the signal
  // It is t -> -cos(20*pi*t)
  auto analyticHilbertSignal = OutputImageType::New();
  analyticHilbertSignal->SetSpacing(spacing);
  analyticHilbertSignal->SetRegions(size);
  analyticHilbertSignal->Allocate();

  itk::ImageRegionIterator<OutputImageType> analyticHilbertSignalIt(analyticHilbertSignal,
                                                                    analyticHilbertSignal->GetLargestPossibleRegion());

  while (!analyticHilbertSignalIt.IsAtEnd())
  {
    ix = analyticHilbertSignalIt.GetIndex();
    double t = ix[0] * spacing[0];
    double v = -cos(20 * itk::Math::pi * t);
    analyticHilbertSignalIt.Set(v);
    ++analyticHilbertSignalIt;
  }

  // Compute the Hilbert transform of the signal using rtkFFTHilbertImageFilter
  using HilbertType = rtk::FFTHilbertImageFilter<OutputImageType>;
  auto hilbert = HilbertType::New();
  hilbert->SetInput(signal);

  HilbertType::ZeroPadFactorsType zeroPadFactors;
  zeroPadFactors.Fill(1);
  hilbert->SetZeroPadFactors(zeroPadFactors);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(hilbert->Update());

  CheckImageQuality<OutputImageType>(hilbert->GetOutput(), analyticHilbertSignal, .035, 27.3, 0.96);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
