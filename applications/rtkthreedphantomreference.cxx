#include "rtkthreedphantomreference_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawQuadricFunctor.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>


int main(int argc, char * argv[])
{
  GGO(rtkthreedphantomreference, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension >                  OutputImageType;

  // Create a stack of empty projection images
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkthreedphantomreference>(constantImageSource, args_info);

  //We create the reference
  typedef rtk::DrawQuadricFunctor<OutputImageType, OutputImageType> DQType;
  if(args_info.verbose_flag)
    std::cout << "Creating reference... " << std::flush;
  DQType::Pointer dq = DQType::New();
  dq->SetNumberOfThreads(1);
  dq->SetInput(constantImageSource->GetOutput());
  dq->SetConfigFile(args_info.phantomfile_arg);
  dq->Update();

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( dq->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Writing reference... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
