#include "rtkMacro.h"
#include "rtkTestConfiguration.h"
#include "itkImageFileReader.h"
#include "rtkAmsterdamShroudImageFilter.h"
#include "rtkConstantImageSource.h"
#include <iomanip>
#include <fstream>

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkProjectGeometricPhantomImageFilter.h"
#include "itkPasteImageFilter.h"
#include "rtkConfiguration.h"

template<class TInputImage>
void CheckImageQuality(typename TInputImage::Pointer recon, typename TInputImage::Pointer ref)
{
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
}

int main(int, char** )
{
  const unsigned int Dimension = 3;
  typedef float                                      OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;
  unsigned int NumberOfProjectionImages = 100;

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
  sizeOutput[0] = 128;
  sizeOutput[1] = 128;
  sizeOutput[2] = 1;
  spacing[0] = 2.5;
  spacing[1] = 2.5;
  spacing[2] = 2.5;
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

  // Amsterdam shroud
  typedef rtk::AmsterdamShroudImageFilter<OutputImageType> shroudFilterType;
  shroudFilterType::Pointer shroudFilter = shroudFilterType::New();
  shroudFilter->SetInput( pasteFilter->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(shroudFilter->Update());

  // Read reference object
  typedef itk::ImageFileReader<  shroudFilterType::OutputImageType > ReaderAmsterdamType;
  ReaderAmsterdamType::Pointer reader2 = ReaderAmsterdamType::New();
  reader2->SetFileName(std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/AmsterdamShroud/Amsterdam.mha"));
  reader2->Update();

  CheckImageQuality< shroudFilterType::OutputImageType >(shroudFilter->GetOutput(), reader2->GetOutput());
  std::cout << "Test PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
