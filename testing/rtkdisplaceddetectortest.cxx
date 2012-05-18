#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkImageRegionConstIterator.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <vcl_cmath.h>

#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawQuadricImageFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkDisplacedDetectorImageFilter.h"

int main(int argc, char* argv[])
{
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // RTK geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(int noProj=0; noProj<360; noProj++)
    {
    geometry->AddProjection(1000.,
                            1536.,
                            noProj,
                            -120.,
                            0.);
    }
  geometry->Update();

  // Constant image sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer tomographySource  = ConstantImageSourceType::New();
  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  ConstantImageSourceType::PointType origin;
  origin.Fill(-127.5);
  ConstantImageSourceType::SizeType size;
  size.Fill(256);
  ConstantImageSourceType::SpacingType spacing;
  spacing.Fill(1.);
  tomographySource->SetOrigin( origin );
  tomographySource->SetSpacing( spacing );
  tomographySource->SetSize( size );
  tomographySource->SetConstant( 0. );
  tomographySource->UpdateOutputInformation();
  size[2]=360;
  projectionsSource->SetOrigin( origin );
  projectionsSource->SetSpacing( spacing );
  projectionsSource->SetSize( size );
  projectionsSource->SetConstant( 0. );
  projectionsSource->UpdateOutputInformation();

  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp=SLPType::New();
  slp->SetInput( projectionsSource->GetOutput() );
  slp->SetGeometry( geometry );
  slp->Update();

  // Displaced detector weighting
  typedef rtk::DisplacedDetectorImageFilter< OutputImageType > DDFType;
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput( slp->GetOutput() );
  ddf->SetGeometry( geometry );

  // FDK reconstruction filtering
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
  FDKCPUType::Pointer feldkamp = FDKCPUType::New();
  feldkamp->SetInput( 0, tomographySource->GetOutput() );
  feldkamp->SetInput( 1, ddf->GetOutput() );
  feldkamp->SetGeometry( geometry );
  feldkamp->GetRampFilter()->SetTruncationCorrection(0.);
  feldkamp->GetRampFilter()->SetHannCutFrequency(0.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() )

  // Create a reference object (in this case a 3D phantom reference).
  typedef rtk::DrawSheppLoganFilter<OutputImageType, OutputImageType> DSLType;
  DSLType::Pointer dsl = DSLType::New();
  dsl->SetInput( tomographySource->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( dsl->Update() )

  typedef itk::ImageRegionConstIterator<OutputImageType> ImageIteratorType;
  ImageIteratorType itTest( feldkamp->GetOutput(), feldkamp->GetOutput()->GetBufferedRegion() );
  ImageIteratorType itRef( dsl->GetOutput(), dsl->GetOutput()->GetBufferedRegion() );

  typedef double ErrorType;
  ErrorType TestError     = 0.;
  ErrorType TestTolerance = 0.005;
  ErrorType EnerError     = 0.;
  bool Exit               = true;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while( !itRef.IsAtEnd() )
    {
    OutputPixelType TestVal = itTest.Get();
    OutputPixelType RefVal = itRef.Get();
    if( TestVal != RefVal )
      {
        TestError += abs(RefVal - TestVal);
        EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.0);
      }
    ++itTest;
    ++itRef;
    }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/(size[0]*size[1]*size[2]);
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError/(size[0]*size[1]*size[2]);
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20*log10(2.0) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (2.0-ErrorPerPixel)/2.0;
  std::cout << "QI = " << QI << std::endl;
  // Checking results
  if (ErrorPerPixel > TestTolerance)
  {
    std::cerr << "Test Failed, Error per pixel not valid!" << std::endl;
    Exit = false;
  }
  if (PSNR < 25.0)
  {
    std::cerr << "Test Failed, PSNR not valid!" << std::endl;
    Exit = false;
  }

  if(Exit)
  {
    std::cerr << "Test PASSED! " << std::endl;
    return EXIT_SUCCESS;
  }
  else
  {
    // Write out the reconstructed image
    typedef itk::ImageFileWriter<  OutputImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( "reconstruction.mha" );
    writer->SetInput( feldkamp->GetOutput() );
    TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

    // Write out the reference image
    writer->SetInput(dsl->GetOutput());
    writer->SetFileName("reference.mha");
    TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

    return EXIT_FAILURE;
  }
}
