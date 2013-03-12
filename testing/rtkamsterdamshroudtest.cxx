#include "rtkMacro.h"
#include "rtkTestConfiguration.h"
#include "itkImageFileReader.h"
#include "rtkAmsterdamShroudImageFilter.h"
#include "rtkConstantImageSource.h"
#include <iomanip>
#include <fstream>

#include "rtkTestConfiguration.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkProjectGeometricPhantomImageFilter.h"
#include "itkPasteImageFilter.h"
#include "rtkConfiguration.h"
#include "rtkReg1DExtractShroudSignalImageFilter.h"
#include "rtkDPExtractShroudSignalImageFilter.h"

template<class TInputImage>
void CheckImageQuality(typename TInputImage::Pointer recon, typename TInputImage::Pointer ref)
{
#if !(FAST_TESTS_NO_CHECKS)
  typedef itk::ImageRegionConstIterator<TInputImage>  ImageIteratorInType;
  ImageIteratorInType itTest( recon, recon->GetBufferedRegion() );
  ImageIteratorInType itRef( ref, ref->GetBufferedRegion() );

  typedef double ErrorType;
  ErrorType TestError = 0.;
  ErrorType EnerError = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while( !itRef.IsAtEnd() )
    {
    typename TInputImage::PixelType TestVal = itTest.Get();
    typename TInputImage::PixelType RefVal  = itRef.Get();

    TestError += vcl_abs(RefVal - TestVal);
    EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.);
    ++itTest;
    ++itRef;
    }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError/ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20*log10(6304.0) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (6304.-ErrorPerPixel)/6304.;
  std::cout << "QI = " << QI << std::endl;

  // Checking results
  if (ErrorPerPixel > 1.20e-6)
  {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 1.20e-6." << std::endl;
    exit( EXIT_FAILURE);
  }
  if (PSNR < 185.)
  {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 185" << std::endl;
    exit( EXIT_FAILURE);
  }
#endif
}

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef double                                     reg1DPixelType;
  typedef float                                      OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;
  typedef itk::Image< reg1DPixelType, Dimension-2 >  reg1DImageType;
#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 100;
#endif

  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometryMain = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometryMain->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);

  // Create a stack of empty projection images
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType sizeOutput;
  ConstantImageSourceType::SpacingType spacing;

  // Adjust size according to geometry and for just one projection
  ConstantImageSourceType::Pointer constantImageSourceSingleProjection = ConstantImageSourceType::New();
  origin[0] = -50.;
  origin[1] = -50.;
  origin[2] = -158.75;
#if FAST_TESTS_NO_CHECKS
  sizeOutput[0] = 4;
  sizeOutput[1] = 4;
  sizeOutput[2] = 1;
  spacing[0] = 106.;
  spacing[1] = 106.;
  spacing[2] = 2.5;
#else
  sizeOutput[0] = 128;
  sizeOutput[1] = 128;
  sizeOutput[2] = 1;
  spacing[0] = 2.5;
  spacing[1] = 2.5;
  spacing[2] = 2.5;
#endif
  constantImageSourceSingleProjection->SetOrigin( origin );
  constantImageSourceSingleProjection->SetSpacing( spacing );
  constantImageSourceSingleProjection->SetSize( sizeOutput );
  constantImageSourceSingleProjection->SetConstant( 0. );

  // Adjust size according to geometry
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  sizeOutput[2] = NumberOfProjectionImages;
  constantImageSource->SetOrigin( origin );
  constantImageSource->SetSpacing( spacing );
  constantImageSource->SetSize( sizeOutput );
  constantImageSource->SetConstant( 0. );

  typedef itk::PasteImageFilter <OutputImageType, OutputImageType, OutputImageType > PasteImageFilterType;
  OutputImageType::IndexType destinationIndex;
  destinationIndex[0] = 0;
  destinationIndex[1] = 0;
  destinationIndex[2] = 0;

  PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();

  // Single projection
  typedef rtk::ProjectGeometricPhantomImageFilter<OutputImageType, OutputImageType> PPCType;
  PPCType::Pointer ppc = PPCType::New();

  std::ofstream      myConfigFile;
  double             size   = 80.;
  double             sinus  = 0.;
  const unsigned int Cycles = 4;

  OutputImageType::Pointer wholeImage = constantImageSource->GetOutput();
  for (unsigned int i=1; i <= sizeOutput[2]; i++)
  {
    // Geometry object
    GeometryType::Pointer geometry = GeometryType::New();
    geometry->AddProjection(1200., 1500., i*360/sizeOutput[2]);
    // Creating phantom config file for each projection
    myConfigFile.open("phantom.txt");
    myConfigFile <<"[Ellipsoid]  A=88.32 B=115.2 C=117.76 x=0 y=0 z=0 beta=0 gray=2.0\n"
                 <<"[Ellipsoid]  A=35. B=" << size - sinus << " C=" << size - sinus << " x=-37. y=0 z=0 beta=0 gray=-1.98\n"
                 <<"[Ellipsoid]  A=35. B=" << size - sinus << " C=" << size - sinus << " x=37.  y=0 z=0 beta=0 gray=-1.98\n"
                 <<"[Ellipsoid]  A=8.  B=8.  C=8.  x=-40. y=0 z=0 beta=0 gray=1.42\n"
                 << std::endl;
    myConfigFile.close();

    // Creating movement
    sinus = 15*sin( i*2*itk::Math::pi/(sizeOutput[2]/Cycles) );
    // Generating projection
    ppc->SetInput(constantImageSourceSingleProjection->GetOutput());
    ppc->SetGeometry(geometry);
    ppc->SetConfigFile("phantom.txt");
    ppc->Update();
    // Adding each projection to volume
    pasteFilter->SetSourceImage(ppc->GetOutput());
    pasteFilter->SetDestinationImage(wholeImage);
    pasteFilter->SetSourceRegion(ppc->GetOutput()->GetLargestPossibleRegion());
    pasteFilter->SetDestinationIndex(destinationIndex);
    pasteFilter->Update();
    wholeImage = pasteFilter->GetOutput();
    destinationIndex[2]++;
  }
  itksys::SystemTools::RemoveFile("phantom.txt");

  std::cout << "\n\n****** Case 1: Amsterdam Shroud Image ******" << std::endl;

  // Amsterdam shroud
  typedef rtk::AmsterdamShroudImageFilter<OutputImageType> shroudFilterType;
  shroudFilterType::Pointer shroudFilter = shroudFilterType::New();
  shroudFilter->SetInput( pasteFilter->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(shroudFilter->Update());

  // Read reference object
  typedef itk::ImageFileReader< shroudFilterType::OutputImageType > ReaderAmsterdamType;
  ReaderAmsterdamType::Pointer reader2 = ReaderAmsterdamType::New();
  reader2->SetFileName(std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/AmsterdamShroud/Amsterdam.mha"));
  reader2->Update();

  CheckImageQuality< shroudFilterType::OutputImageType >(shroudFilter->GetOutput(), reader2->GetOutput());
  std::cout << "Test PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: Breathing signal calculated by reg1D algorithm ******\n" << std::endl;

  //Estimation of breathing signal
  typedef rtk::Reg1DExtractShroudSignalImageFilter< reg1DPixelType, reg1DPixelType > reg1DFilterType;
  reg1DImageType::Pointer  reg1DSignal;
  reg1DFilterType::Pointer reg1DFilter = reg1DFilterType::New();
  reg1DFilter->SetInput( reader2->GetOutput() );
  reg1DFilter->Update();
  reg1DSignal = reg1DFilter->GetOutput();

#if !(FAST_TESTS_NO_CHECKS)
  //Test Reference
  float reg1D[100] = {0, 4, 7.625, 10.75, 13.25, 15, 15.75, 15.625, 14.5, 12.625, 9.875, 6.5, 2.875, -1, -5.5, -9.625, -13.625, -16.5,
  -18.625, -19.5, -19, -17.25, -14.375, -10.75, -6.375, -1.875, 2.5, 6.625, 10.25, 13.25, 15.25, 16.375, 16.375, 15.25, 13.375,
  10.625, 7.25, 3.625, -0, -3.625, -6.375, -8.75, -10.375, -12.25, -13.625, -13.625, -12.125, -9.5, -6.25, -2.625, 1.5, 5.625, 9.375,
  12.5, 15, 16.75, 17.5, 17.375, 16.25, 14.375, 11.625, 8.25, 4.625, 0.75, -3.75, -7.875, -11.875, -14.75, -16.875, -17.75,
  -17.25, -15.5, -12.625, -9, -4.625, -0.125, 4.25, 8.375, 12, 15, 17, 18.125, 18.125, 17, 15.125, 12.375, 9, 5.375, 1.75,
  -1.875, -4.625, -7, -8.625, -10.5, -11.875, -11.875, -10.375, -7.75, -4.5, -0.875};

  //Checking for possible errors
  float zeroValue = 1e-12;
  float sum = 0.;
  unsigned int i = 0;
  itk::ImageRegionConstIterator<reg1DImageType> it( reg1DSignal, reg1DSignal->GetLargestPossibleRegion() );
  for (it.GoToBegin(); !it.IsAtEnd(); ++it, i++)
  {
    sum += vcl_abs(reg1D[i] - it.Get());
  }

  if ( sum <= zeroValue )
    std::cout << "Test PASSED! " << std::endl;
  else
  {
    std::cerr << "Test FAILED! " << "Breathing signal does not match, absolute difference " << sum << " instead of 0." << std::endl;
    exit( EXIT_FAILURE);
  }
#endif

  std::cout << "\n\n****** Case 3: Breathing signal calculated by DP algorithm ******\n" << std::endl;

  //Estimation of breathing signal
  typedef rtk::DPExtractShroudSignalImageFilter< reg1DPixelType, reg1DPixelType > DPFilterType;
  reg1DImageType::Pointer  DPSignal;
  DPFilterType::Pointer DPFilter = DPFilterType::New();
  DPFilter->SetInput( reader2->GetOutput() );
  DPFilter->SetAmplitude( 20. );
  DPFilter->Update();
  DPSignal = DPFilter->GetOutput();

#if !(FAST_TESTS_NO_CHECKS)
  //Test Reference
  float DP[100] = {5, 7.5, 12.5, 17.5, 20, 22.5, 22.5, 22.5, 20, 17.5, 15, 12.5, 7.5, 2.5, -2.5, -5, -10, -12.5, -12.5, -15, -12.5, -12.5, -7.5, -5,
  0, 5, 7.5, 12.5, 17.5, 20, 22.5, 22.5, 22.5, 20, 20, 15, 12.5, 7.5, 2.5, -2.5, -5, -10, -12.5, -12.5, -12.5, -12.5, -10, -7.5,
  -5, -0, 5, 7.5, 12.5, 17.5, 20, 22.5, 22.5, 22.5, 20, 17.5, 15, 12.5, 7.5, 2.5, -2.5, -5, -10, -12.5, -12.5, -15, -12.5, -12.5,
  -7.5, -5, -0, 5, 7.5, 12.5, 17.5, 20, 22.5, 22.5, 22.5, 20, 20, 15, 12.5, 7.5, 2.5, -2.5, -5, -10, -12.5, -12.5, -12.5, -12.5,
  -10, -7.5, -5, -0};

  //Checking for possible errors
  sum = 0.;
  i = 0;
  itk::ImageRegionConstIterator< reg1DImageType > itDP( DPSignal, DPSignal->GetLargestPossibleRegion() );
  for (itDP.GoToBegin(); !itDP.IsAtEnd(); ++itDP, i++)
  {
    sum += vcl_abs(DP[i] - itDP.Get());
  }

  if ( sum <= zeroValue )
    std::cout << "Test PASSED! " << std::endl;
  else
  {
    std::cerr << "Test FAILED! " << "Breathing signal does not match, absolute difference " << sum << " instead of 0." << std::endl;
    exit( EXIT_FAILURE);
  }
#endif

  return EXIT_SUCCESS;
}
