#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
//#include <iostream>
#include "rtkthreedphantomreferencetest_ggo.h"
#include "rtkGgoFunctions.h"
#include <fstream>
#include "itkRandomImageSource.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkRawImageIO.h"
#include "itkImageRegionConstIterator.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawQuadricFunctor.h"

#include "rtkProjectionsReader.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include <itkRegularExpressionSeriesFileNames.h>

int main(int argc, char* argv[])
{
  GGO(rtkthreedphantomreferencetest, args_info);
  const unsigned int Dimension = 3;
  typedef bool                                                           BooleanType;
  typedef float                                                          OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension >                       OutputImageType;
  typedef OutputImageType::PixelType                                     PixelType;
  typedef double                                                         ErrorType;
  typedef rtk::DrawQuadricFunctor<OutputImageType, OutputImageType>      DQType;
  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  typedef itk::ImageRegionConstIterator<OutputImageType>                 ImageIteratorType;
  typedef rtk::ConstantImageSource< OutputImageType >                    ConstantImageSourceType;

  // Geometry
  // RTK geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Projection matrices
  for(int noProj=0; noProj<360; noProj++)
    {
    double angle = 0. + noProj * 360. / 360;
    geometry->AddProjection(1000.,
                            1536.,
                            angle,
                            0.,
                            0.);
    }
  geometry->Update();

  // Constant Image Sources whether for the reference and test objects
  ConstantImageSourceType::Pointer constantImageSourceRef = ConstantImageSourceType::New();
  // Constant Image Source for the reference object, using ggo parameters
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkthreedphantomreferencetest>(constantImageSourceRef, args_info);

  ConstantImageSourceType::SizeType sizeOutput;
  sizeOutput[0] = constantImageSourceRef->GetSize()[0];
  sizeOutput[1] = constantImageSourceRef->GetSize()[1];
  sizeOutput[2] = geometry->GetGantryAngles().size();
  constantImageSourceRef->SetSize( sizeOutput );

  typedef rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  SLPType::Pointer slp=SLPType::New();
  slp->SetInput(constantImageSourceRef->GetOutput());
  slp->SetGeometry(geometry);
  slp->SetConfigFile(args_info.phantomfile_arg);
  slp->Update();

#define SET_FELDKAMP_OPTIONS(f) \
    f->SetInput( 0, constantImageSourceRef->GetOutput() ); \
    f->SetInput( 1, slp->GetOutput() ); \
    f->SetGeometry( geometry ); \
    f->GetRampFilter()->SetTruncationCorrection(args_info.pad_arg); \
    f->GetRampFilter()->SetHannCutFrequency(args_info.hann_arg);

  // FDK reconstruction filtering
  itk::ImageToImageFilter<OutputImageType, OutputImageType>::Pointer feldkamp;
  typedef rtk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
  feldkamp = FDKCPUType::New();
  SET_FELDKAMP_OPTIONS( static_cast<FDKCPUType*>(feldkamp.GetPointer()) );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() )

  // Create a reference object (in this case a 3D phantom reference).
  DQType::Pointer dq = DQType::New();
  dq->SetInput(constantImageSourceRef->GetOutput());
  dq->SetConfigFile(args_info.phantomfile_arg);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( dq->Update() )

  ImageIteratorType itTest( feldkamp->GetOutput(), feldkamp->GetOutput()->GetBufferedRegion() );
  ImageIteratorType itRef( dq->GetOutput(), dq->GetOutput()->GetBufferedRegion() );

  ErrorType TestError     = 0.;
  ErrorType TestTolerance = 0.005;
  ErrorType EnerError     = 0.;
  BooleanType Exit        = true;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while( !itRef.IsAtEnd() )
    {
    PixelType TestVal = itTest.Get();
    PixelType RefVal = itRef.Get();
    if( TestVal != RefVal )
      {
        TestError += abs(RefVal - TestVal);
        EnerError += pow((RefVal - TestVal), 2.0);
      }
    ++itTest;
    ++itRef;
    }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/pow(args_info.dimension_arg[0], 3.0);
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError/pow(args_info.dimension_arg[0], 3.0);
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
    // Write out the TEST object
    typedef itk::ImageFileWriter<  OutputImageType > WriterType;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( args_info.output_arg[1] );
    writer->SetInput( feldkamp->GetOutput() );

    TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

    // Write out the REFERENCE object
    writer->SetInput(dq->GetOutput());
    writer->SetFileName(args_info.output_arg[0]);

    TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )
    return EXIT_FAILURE;
  }
}




