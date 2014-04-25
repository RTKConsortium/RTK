
#include "rtkTestConfiguration.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"

// ITK includes
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

#ifdef USE_CUDA
#  include "rtkCudaForwardProjectionImageFilter.h"
#else
#  include "rtkJosephForwardProjectionImageFilter.h"
#endif

void WriteImages()
{
  std::string name = "C:/img5";

  const unsigned int Dimension = 3;
  typedef float OutputPixelType;

#ifdef USE_CUDA
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#endif

  const unsigned int NumberOfProjectionImages = 45;

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  // Create Joseph Forward Projector volume input.
  const ConstantImageSourceType::Pointer volInput = ConstantImageSourceType::New();
  origin[0] = -126;
  origin[1] = -126;
  origin[2] = -126;
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
  volInput->SetOrigin( origin );
  volInput->SetSpacing( spacing );
  volInput->SetSize( size );
  volInput->SetConstant( 1. );
  volInput->UpdateOutputInformation();

  // Initialization Volume, it is used in the Joseph Forward Projector and in the
  // Ray Box Intersection Filter in order to initialize the stack of projections.
  const ConstantImageSourceType::Pointer projInput = ConstantImageSourceType::New();
  size[2] = NumberOfProjectionImages;
  projInput->SetOrigin( origin );
  projInput->SetSpacing( spacing );
  projInput->SetSize( size );
  projInput->SetConstant( 0. );
  projInput->Update();

  // Joseph Forward Projection filter
#ifdef USE_CUDA
  typedef rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType> JFPType;
#else
  typedef rtk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType> JFPType;
#endif
  JFPType::Pointer jfp = JFPType::New();
  jfp->InPlaceOff();
  jfp->SetInput( projInput->GetOutput() );
  jfp->SetInput( 1, volInput->GetOutput() );


  std::cout << "\n\n****** Case 1: inner ray source ******" << std::endl;
  // Geometry
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int i=0; i<NumberOfProjectionImages;i++)
    {
    const double angle = -45. + i*2.;
    geometry->AddProjection(47.6 / vcl_cos(angle*itk::Math::pi/180.), 1000., angle);
    }
  jfp->SetGeometry(geometry);
  jfp->Update();
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( name + "_case1.mha" );
  writer->SetInput( jfp->GetOutput() );
  writer->Update();
  std::cout << "\n\nWriten ! " << std::endl;

  std::cout << "\n\n****** Case 2: outer ray source ******" << std::endl;
  // Geometry
  geometry = GeometryType::New();
  for(unsigned int i=0; i<NumberOfProjectionImages; i++)
    geometry->AddProjection(500., 1000., i*8.);
  jfp->SetGeometry( geometry );
  jfp->Update();
  writer->SetFileName( name + "_case2.mha" );
  writer->SetInput( jfp->GetOutput() );
  writer->Update();
  std::cout << "\n\nWriten ! " << std::endl;

  std::cout << "\n\n****** Case 3: Shepp-Logan, outer ray source ******" << std::endl;
  // Create a Shepp Logan reference volume (finer resolution)
  origin.Fill(-127);
  size.Fill(128);
  spacing.Fill(2.);
  volInput->SetOrigin( origin );
  volInput->SetSpacing( spacing );
  volInput->SetSize( size );
  volInput->SetConstant( 0. );
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl = DSLType::New();
  dsl->InPlaceOff();
  dsl->SetInput( volInput->GetOutput() );
  dsl->Update();
  // Forward projection
  jfp->SetInput( 1, dsl->GetOutput() );
  jfp->Update();
  writer->SetFileName( name + "_case3.mha" );
  writer->SetInput( jfp->GetOutput() );
  writer->Update();
  std::cout << "\n\nWriten ! " << std::endl;

  std::cout << "\n\n****** Case 4: Shepp-Logan, inner ray source ******" << std::endl;
  geometry = GeometryType::New();
  for(unsigned int i=0; i<NumberOfProjectionImages; i++)
    geometry->AddProjection(120., 1000., i*8.);
  jfp->SetGeometry( geometry );
  jfp->Update();
  writer->SetFileName( name + "_case4.mha" );
  writer->SetInput( jfp->GetOutput() );
  writer->Update();
  std::cout << "\n\nWriten ! " << std::endl;
}

void CompareImages()
{
  const unsigned int Dimension = 3;
  typedef float OutputPixelType;

  typedef itk::Image< OutputPixelType, Dimension > ImageType;
  typedef itk::ImageRegionConstIterator<ImageType> ImageIteratorType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  
  std::string ref1 = "C:/data/img4_case";
  std::string test1 = "C:/data/img5_case";

  std::string values[4] = { "1", "2", "3", "4" };

  for(int i = 0; i < 4; i++)
    {

    ReaderType::Pointer readerRef = ReaderType::New();
    readerRef->SetFileName( ref1 + values[i] + ".mha" );
    readerRef->Update();
    ImageIteratorType itRef( readerRef->GetOutput(), readerRef->GetOutput()->GetBufferedRegion() );
  
    ReaderType::Pointer readerTest = ReaderType::New();
    readerTest->SetFileName( test1 + values[i] + ".mha" );
    readerTest->Update();
    ImageIteratorType itTest( readerTest->GetOutput(), readerTest->GetOutput()->GetBufferedRegion() );

    typedef double ErrorType;
    ErrorType TestError = 0.;
    ErrorType EnerError = 0.;

    itTest.GoToBegin();
    itRef.GoToBegin();

    while( !itRef.IsAtEnd() )
      {
      ImageType::PixelType TestVal = itTest.Get();
      ImageType::PixelType RefVal = itRef.Get();

      if( TestVal != RefVal )
        {
        TestError += vcl_abs(RefVal - TestVal);
        EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.);
        }
      ++itTest;
      ++itRef;
      }

    // Error per Pixel
    ErrorType ErrorPerPixel = TestError/readerTest->GetOutput()->GetBufferedRegion().GetNumberOfPixels();
    std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
    // MSE
    ErrorType MSE = EnerError/readerRef->GetOutput()->GetBufferedRegion().GetNumberOfPixels();
    std::cout << "MSE = " << MSE << std::endl;
    // PSNR
    ErrorType PSNR = 20*log10(255.0) - 10*log10(MSE);
    std::cout << "PSNR = " << PSNR << "dB" << std::endl;
    // QI
    ErrorType QI = (255.0-ErrorPerPixel)/255.0;
    std::cout << "QI = " << QI << std::endl;

    // Checking results
    if (ErrorPerPixel > 1.28)
      {
      std::cerr << "Test Error per pixel Failed! "
                << ErrorPerPixel << " instead of 1.28" << std::endl;
      }
    if (PSNR < 44.)
      {
      std::cerr << "Test PSNR Failed! "
                << PSNR << " instead of 44" << std::endl;
      }
    }
}

int main(int , char** )
{
  //WriteImages();
  CompareImages();
  getchar();

  return EXIT_SUCCESS;
}
