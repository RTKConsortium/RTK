
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkDrawSheppLoganFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "rtkJosephForwardProjectionImageFilter.h"
#include <itkExtractImageFilter.h>
#include <itkMultiplyByConstantImageFilter.h>
#include <rtkSheppLoganPhantomFilter.h>

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
  if (ErrorPerPixel > 0.5)
  {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 0.005" << std::endl;
    exit( EXIT_FAILURE);
  }
  if (PSNR < 25.)
  {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 25" << std::endl;
    exit( EXIT_FAILURE);
  }
}

int main(int , char** )
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::Vector<double, 3>                   VectorType;
  const unsigned int NumberOfProjectionImages = 1;

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType size;
  ConstantImageSourceType::SpacingType spacing;

  // Resized Volume, it is used as the reference to calculate the Joseph Forward Projector quality.
  const ConstantImageSourceType::Pointer resizedInput = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 255;
  size[1] = 255;
  size[2] = 175;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  resizedInput->SetOrigin( origin );
  resizedInput->SetSpacing( spacing );
  resizedInput->SetSize( size );
  resizedInput->SetConstant( 1. );

  // Full Volume, it is used in the Joseph Forward Projector in order to check
  // the case of an inner ray source.
  const ConstantImageSourceType::Pointer fullInput = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 255;
  size[1] = 255;
  size[2] = 255;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  fullInput->SetOrigin( origin );
  fullInput->SetSpacing( spacing );
  fullInput->SetSize( size );
  fullInput->SetConstant( 1. );

  // Initialization Volume, it is used in the Joseph Forward Projector and in the
  // Ray Box Intersection Filter in order to initialize the stack of projections.
  const ConstantImageSourceType::Pointer initInput = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 255;
  size[1] = 255;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  initInput->SetOrigin( origin );
  initInput->SetSpacing( spacing );
  initInput->SetSize( size );
  initInput->SetConstant( 0. );

  // Shepp-Logan Full Volume
  const ConstantImageSourceType::Pointer sheppFullInput = ConstantImageSourceType::New();
  origin[0] = -127;
  origin[1] = -127;
  origin[2] = -127;
  size[0] = 255;
  size[1] = 255;
  size[2] = 255;
  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  sheppFullInput->SetOrigin( origin );
  sheppFullInput->SetSpacing( spacing );
  sheppFullInput->SetSize( size );
  sheppFullInput->SetConstant( 0. );

  // Ray Box Intersection filter
  typedef rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType> RBIType;
  RBIType::Pointer rbi = RBIType::New();

  // Joseph Forward Projection filter
  typedef rtk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType> JFPType;
  JFPType::Pointer jfp = JFPType::New();

//  // Writer
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();

  std::cout << "\n\n****** Case 1: inner ray source ******" << std::endl;
  // Geometry
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  // Creating geometry
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(47.5, 1000., noProj);

  // Calculating ray box intersection of the resized Input
  resizedInput->UpdateOutputInformation();
  rbi->SetInput( initInput->GetOutput() );
  rbi->SetBoxFromImage( resizedInput->GetOutput() );
  rbi->SetGeometry( geometry );
  rbi->Update();

  // Calculating Joseph forward projection of the full Input with an inner ray source
  jfp->SetInput(initInput->GetOutput());
  jfp->SetInput( 1, fullInput->GetOutput() );
  jfp->SetGeometry( geometry );
  jfp->Update();

  CheckImageQuality<OutputImageType>(rbi->GetOutput(), jfp->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 2: inner ray source, 90 degrees and source not in the middle of a pixel ******" << std::endl;
  // Geometry
  geometry = GeometryType::New();
  // Creating geometry
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(47.3, 1000., noProj+90);
  // Creating the Box
  VectorType boxMin, boxMax;
  boxMin[0] = -127.;
  boxMin[1] = -127.;
  boxMin[2] = -127.;
  boxMax[0] = boxMin[0]+174.8;
  boxMax[1] = boxMin[1]+255.;
  boxMax[2] = boxMin[2]+255.;

  // Calculating ray box intersection of the resized Input
  resizedInput->UpdateOutputInformation();
  rbi->SetInput( initInput->GetOutput() );
  rbi->SetBoxMin(boxMin);
  rbi->SetBoxMax(boxMax);
  rbi->SetGeometry( geometry );
  rbi->Update();

  // Calculating Joseph forward projection of the full Input with an inner ray source
  jfp->SetInput(initInput->GetOutput());
  jfp->SetInput( 1, fullInput->GetOutput() );
  jfp->SetGeometry( geometry );
  jfp->SetNumberOfThreads(1);
  jfp->Update();

  CheckImageQuality<OutputImageType>(rbi->GetOutput(), jfp->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 3: outer ray source ******" << std::endl;
  // Geometry
  geometry = GeometryType::New();
  // Creating geometry
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(128.5, 1000., noProj);
  // We resize resizedInput, in order to have a full Input 255x255x255
  size[0] = 255;
  size[1] = 255;
  size[2] = 255;
  resizedInput->SetSize( size );

  // Calculating ray box intersection of the resized Input
  resizedInput->UpdateOutputInformation();
  rbi->SetInput( initInput->GetOutput() );
  rbi->SetBoxFromImage( resizedInput->GetOutput() );
  rbi->SetGeometry( geometry );
  rbi->Update();

  // Calculating Joseph forward projection of the full Input with an inner ray source
  jfp->SetInput(initInput->GetOutput());
  jfp->SetInput( 1, fullInput->GetOutput() );
  jfp->SetGeometry( geometry );
  jfp->Update();

  CheckImageQuality<OutputImageType>(rbi->GetOutput(), jfp->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  std::cout << "\n\n****** Case 4: Shepp-Logan, outer ray source ******" << std::endl;
  // Geometry
  geometry = GeometryType::New();
  // Creating geometry
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(120, 256., noProj);

  // Create a reference object (in this case a 3D phantom reference).
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl = DSLType::New();

  size[0] = 255;
  size[1] = 255;
  size[2] = 1; //Number of Projections
  initInput->SetSize( size );
  initInput->UpdateOutputInformation();

  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp=SLPType::New();
  slp->SetInput(initInput->GetOutput());
  slp->SetGeometry(geometry);
  slp->SetPhantomScale(128.);
  slp->Update();

  dsl = DSLType::New();
  dsl->SetInput( sheppFullInput->GetOutput() );
  dsl->SetPhantomScale(128.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )

  // Calculating Joseph forward projection of the full Input with an inner ray source
  jfp->SetInput(initInput->GetOutput());
  jfp->SetInput( 1, dsl->GetOutput() );
  jfp->SetGeometry( geometry );
  jfp->Update();

  CheckImageQuality<OutputImageType>(slp->GetOutput(), jfp->GetOutput());
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
