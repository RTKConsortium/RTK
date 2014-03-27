#include "rtkadmmtotalvariation_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include <itkTimeProbe.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include "rtkProjectionsReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkConstantImageSource.h"
#include "rtkDisplacedDetectorImageFilter.h"
#include "rtkADMMTotalVariationConeBeamReconstructionFilter.h"

int main(int argc, char * argv[])
{
  GGO(rtkadmmtotalvariation, args_info);

  typedef float OutputPixelType;
  typedef itk::Image< OutputPixelType, 3 > OutputImageType;

  //////////////////////////////////////////////////////////////////////////////////////////
  // Read all the inputs
  //////////////////////////////////////////////////////////////////////////////////////////

  // Read projections
  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(false);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(0);

  if(args_info.verbose_flag)
      std::cout << "Regular expression matches "
                << names->GetFileNames().size()
                << " file(s)..."
                << std::endl;

  // Projections reader
  typedef rtk::ProjectionsReader< OutputImageType > projectionsReaderType;
  projectionsReaderType::Pointer projectionsReader = projectionsReaderType::New();
  projectionsReader->SetFileNames( names->GetFileNames() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( projectionsReader->GenerateOutputInformation() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( projectionsReader->Update() );

  // Geometry
  if(args_info.verbose_flag)
      std::cout << "Reading geometry information from "
                << args_info.geometry_arg
                << "..."
                << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() );

  // Displaced detector weighting
  typedef rtk::DisplacedDetectorImageFilter< OutputImageType > DDFType;
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput( projectionsReader->GetOutput() );
  ddf->SetGeometry( geometryReader->GetOutputObject() );

  // Create input: either an existing volume read from a file or a blank image
  itk::ImageSource< OutputImageType >::Pointer inputFilter;
  if(args_info.input_given)
    {
    // Read an existing image to initialize the volume
    typedef itk::ImageFileReader<  OutputImageType > InputReaderType;
    InputReaderType::Pointer inputReader = InputReaderType::New();
    inputReader->SetFileName( args_info.input_arg );
    inputFilter = inputReader;
    }
  else
    {
    // Create new empty volume
    typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkadmmtotalvariation>(constantImageSource, args_info);
    inputFilter = constantImageSource;
    }

  //////////////////////////////////////////////////////////////////////////////////////////
  // Setup the ADMM filter and run it
  //////////////////////////////////////////////////////////////////////////////////////////

  // Set filter to be fed to the conjugate gradient optimizer
  typedef rtk::ADMMTotalVariationConeBeamReconstructionFilter<OutputImageType> ADMM_TV_FilterType;
  ADMM_TV_FilterType::Pointer admmFilter = ADMM_TV_FilterType::New();

  // Set the forward and back projection filters to be used
  admmFilter->ConfigureForwardProjection(args_info.method_arg);
  admmFilter->ConfigureBackProjection(args_info.bp_arg);

  // Set the geometry and interpolation weights
  admmFilter->SetGeometry(geometryReader->GetOutputObject());

  // Set whether or not time probes should be activated
  if (args_info.time_flag)
    {
    admmFilter->SetMeasureExecutionTimes(true);
    }

  // Set all four numerical parameters
  admmFilter->SetCG_iterations(args_info.CGiter_arg);
  admmFilter->SetAL_iterations(args_info.iterations_arg);
  admmFilter->Setalpha(args_info.alpha_arg);
  admmFilter->Setbeta(args_info.beta_arg);

  // Set the inputs of the ADMM filter
  admmFilter->SetInput(0, inputFilter->GetOutput() );
  admmFilter->SetInput(1, projectionsReader->GetOutput() );

  itk::TimeProbe timeProbe;
    if (args_info.time_flag)
    {
    std::cout << "Starting global probes before updating the ADMM filter" << std::endl;
    timeProbe.Start();
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION( admmFilter->Update() );

  if (args_info.time_flag)
    {
    timeProbe.Stop();
    std::cout << "Updating the ADMM filter took " << timeProbe.GetTotal() << ' ' << timeProbe.GetUnit() << std::endl;
    }

  // Set writer and write the output
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( admmFilter->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
