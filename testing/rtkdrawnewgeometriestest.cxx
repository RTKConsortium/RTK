#include "rtkTestConfiguration.h"
#include "rtkMacro.h"
#include "rtkConstantImageSource.h"
#include "rtkGeometricPhantomFileReader.h"
#include "rtkDrawGeometricPhantomImageFilter.h"

typedef rtk::ThreeDCircularProjectionGeometry GeometryType;

template<class TImage>
void CheckImageQuality(typename TImage::Pointer recon, typename TImage::Pointer ref)
{
  typedef itk::ImageRegionConstIterator<TImage> ImageIteratorType;
  ImageIteratorType itTest( recon, recon->GetBufferedRegion() );
  ImageIteratorType itRef( ref, ref->GetBufferedRegion() );

  typedef double ErrorType;
  ErrorType TestError = 0.;
  ErrorType EnerError = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while( !itRef.IsAtEnd() )
    {
    typename TImage::PixelType TestVal = itTest.Get();
    typename TImage::PixelType RefVal = itRef.Get();

    if( TestVal != RefVal )
      {
      TestError += vcl_abs(RefVal - TestVal);
      EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.);
      }
    ++itTest;
    ++itRef;
    }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/recon->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError/ref->GetBufferedRegion().GetNumberOfPixels();
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20*log10(255.0) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (255.0-ErrorPerPixel)/255.0;
  std::cout << "QI = " << QI << std::endl;

  // Checking results
  if (ErrorPerPixel > 1e-10)
    {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 1e-10" << std::endl;
    exit( EXIT_FAILURE);
    }
  if (PSNR < 125.)
    {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 125dB" << std::endl;
    exit( EXIT_FAILURE);
    }
}

int main(int, char** )
{
    const unsigned int Dimension = 3;
    typedef float                                    OutputPixelType;
    typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

    // Constant image sources
    typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
    ConstantImageSourceType::PointType origin;
    ConstantImageSourceType::SizeType size;
    ConstantImageSourceType::SpacingType spacing;

    ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
    origin[0] = -127.;
    origin[1] = -127.;
    origin[2] = -127.;
    size[0] = 128;
    size[1] = 128;
    size[2] = 128;
    spacing[0] = 2.;
    spacing[1] = 2.;
    spacing[2] = 2.;
    tomographySource->SetOrigin( origin );
    tomographySource->SetSpacing( spacing );
    tomographySource->SetSize( size );
    tomographySource->SetConstant( 0. );

    // New Geometries from Configuration File
    typedef rtk::DrawGeometricPhantomImageFilter<OutputImageType, OutputImageType> DGPType;
    DGPType::Pointer dgp=DGPType::New();
    dgp->SetInput( tomographySource->GetOutput() );
    dgp->SetConfigFile(std::string(RTK_DATA_ROOT) +
                       std::string("/Input/GeometricPhantom/Geometries.txt"));
    dgp->Update();

    // Create Reference
    std::vector< double > Axis;
    std::vector< double > Center;

    Axis.push_back(100.);
    Axis.push_back(0.);
    Axis.push_back(100.);
    Center.push_back(2.);
    Center.push_back(2.);
    Center.push_back(2.);

    // Draw CYLINDER
    typedef rtk::DrawCylinderImageFilter<OutputImageType, OutputImageType> DCType;
    DCType::Pointer dcl = DCType::New();
    dcl->SetInput( tomographySource->GetOutput() );
    dcl->SetAxis(Axis);
    dcl->SetCenter(Center);
    dcl->SetAngle(0.);
    dcl->SetAttenuation(2.);
    dcl->Update();

    // Draw CONE
    Axis.clear();
    Axis.push_back(25.);
    Axis.push_back(-50.);
    Axis.push_back(25.);

    typedef rtk::DrawConeImageFilter<OutputImageType, OutputImageType> DCOType;
    typename DCOType::Pointer dco = DCOType::New();
    dco->SetInput( tomographySource->GetOutput() );
    dco->SetAxis(Axis);
    dco->SetCenter(Center);
    dco->SetAngle(0.);
    dco->SetAttenuation(-0.54);
    dco->Update();

    //Add Image Filter used to concatenate the different figures obtained on each iteration
    typedef itk::AddImageFilter <OutputImageType, OutputImageType, OutputImageType> AddImageFilterType;
    typename AddImageFilterType::Pointer addFilter = AddImageFilterType::New();

    addFilter->SetInput1(dcl->GetOutput());
    addFilter->SetInput2(dco->GetOutput());
    addFilter->Update();

    CheckImageQuality<OutputImageType>(dgp->GetOutput(), addFilter->GetOutput());
    std::cout << "Test PASSED! " << std::endl;

    return EXIT_SUCCESS;
}
