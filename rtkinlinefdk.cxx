#include "rtkinlinefdk_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkProjectionsReader.h"
#include "itkDisplacedDetectorImageFilter.h"
#include "itkParkerShortScanImageFilter.h"
#include "itkFDKConeBeamReconstructionFilter.h"
#if CUDA_FOUND
# include "itkCudaFDKConeBeamReconstructionFilter.h"
#endif
#if OPENCL_FOUND
# include "itkOpenCLFDKConeBeamReconstructionFilter.h"
#endif

#include <itkRegularExpressionSeriesFileNames.h>
#include <itkImageFileWriter.h>
#include <itkSimpleFastMutexLock.h>
#include <itkMultiThreader.h>
#include <itksys/SystemTools.hxx>

#include "itkNumericTraits.h"

// Pass projection name, projection parameters, last
struct ThreadInfoStruct
  {
  itk::SimpleFastMutexLock mutex;
  args_info_rtkinlinefdk *args_info;
  bool stop;
  unsigned int nproj;
  double sid;
  double sdd;
  double gantryAngle;
  double projOffsetX;
  double projOffsetY;
  double outOfPlaneAngle;
  double inPlaneAngle;
  double sourceOffsetX;
  double sourceOffsetY;
  double minimumOffsetX;    // Used for Wang weighting
  double maximumOffsetX;
  std::string fileName;
  };

void computeOffsetsFromGeometry(itk::ThreeDCircularProjectionGeometry::Pointer geometry, double *minOffset, double *maxOffset);
static ITK_THREAD_RETURN_TYPE AcquisitionCallback(void *arg);
static ITK_THREAD_RETURN_TYPE InlineThreadCallback(void *arg);

int main(int argc, char * argv[])
{
  GGO(rtkinlinefdk, args_info);

  // Launch threads, one for acquisition, one for reconstruction with inline processing
  ThreadInfoStruct threadInfo;
  threadInfo.args_info = &args_info;
  threadInfo.nproj = 0;
  threadInfo.minimumOffsetX = 0.0;
  threadInfo.maximumOffsetX = 0.0;

  itk::MultiThreader::Pointer threader = itk::MultiThreader::New();
  threader->SetMultipleMethod(0, AcquisitionCallback, (void*)&threadInfo);
  threader->SetMultipleMethod(1, InlineThreadCallback, (void*)&threadInfo);
  threader->SetNumberOfThreads(2);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( threader->MultipleMethodExecute () );

  return EXIT_SUCCESS;
}

// This thread reads in a geometry file and a sequence of projection file names
// and communicates them one by one to the other thread via a ThreadinfoStruct.
static ITK_THREAD_RETURN_TYPE AcquisitionCallback(void *arg)
{
  double minOffset, maxOffset;
  ThreadInfoStruct *threadInfo = (ThreadInfoStruct *)(((itk::MultiThreader::ThreadInfoStruct *)(arg))->UserData);

  threadInfo->mutex.Lock();

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(threadInfo->args_info->path_arg);
  names->SetNumericSort(false);
  names->SetRegularExpression(threadInfo->args_info->regexp_arg);
  names->SetSubMatch(0);

  if(threadInfo->args_info->verbose_flag)
    std::cout << "Regular expression matches "
              << names->GetFileNames().size()
              << " file(s)..."
              << std::endl;

  // Geometry
  if(threadInfo->args_info->verbose_flag)
    std::cout << "Reading geometry information from "
              << threadInfo->args_info->geometry_arg
              << "..."
              << std::endl;
  itk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = itk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(threadInfo->args_info->geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() );

  // Computes the minimum and maximum offsets from Geometry
  computeOffsetsFromGeometry(geometryReader->GetOutputObject(), &minOffset, &maxOffset);
  std::cout << " main :"<<  minOffset << " "<< maxOffset <<std::endl;

  threadInfo->mutex.Unlock();

  // Mock an inline acquisition
  unsigned int nproj = geometryReader->GetOutputObject()->GetMatrices().size();
  itk::ThreeDCircularProjectionGeometry *geometry = geometryReader->GetOutputObject();
  for(unsigned int i=0; i<nproj; i++)
    {
    threadInfo->mutex.Lock();
    threadInfo->sdd = geometry->GetSourceToDetectorDistances()[i];
    threadInfo->sid = geometry->GetSourceToIsocenterDistances()[i];
    threadInfo->gantryAngle = geometry->GetGantryAngles()[i];
    threadInfo->sourceOffsetX = geometry->GetSourceOffsetsX()[i];
    threadInfo->sourceOffsetY = geometry->GetSourceOffsetsY()[i];
    threadInfo->projOffsetX = geometry->GetProjectionOffsetsX()[i];
    threadInfo->projOffsetY = geometry->GetProjectionOffsetsY()[i];
    threadInfo->inPlaneAngle = geometry->GetInPlaneAngles()[i];
    threadInfo->outOfPlaneAngle = geometry->GetOutOfPlaneAngles()[i];
    threadInfo->minimumOffsetX = minOffset;
    threadInfo->maximumOffsetX = maxOffset;
    threadInfo->fileName = names->GetFileNames()[ vnl_math_min( i, (unsigned int)names->GetFileNames().size()-1 ) ];
    threadInfo->nproj = i+1;
    threadInfo->stop = (i==nproj-1);
    if(threadInfo->args_info->verbose_flag)
      std::cout << std::endl
                << "AcquisitionCallback has simulated the acquisition of projection #" << i
                << std::endl;
    threadInfo->mutex.Unlock();
    itksys::SystemTools::Delay(200);
    }

  return ITK_THREAD_RETURN_VALUE;
}

// This thread receives information of each projection (one-by-one) and process
// directly the projections for which it has enough information. This thread
// currently assumes that the projections are sequentially sent with increasing
// gantry angles. Specific management with a queue must be implemented if the
// projections are not exactly sequential. Short scans has not been implemented yet because this filter
// currently require the full geometry of the acquisition. Management with a mock geometry file
// would be possible but it is still to be implemented.
static ITK_THREAD_RETURN_TYPE InlineThreadCallback(void *arg)
{
  ThreadInfoStruct *threadInfo = (ThreadInfoStruct *)(((itk::MultiThreader::ThreadInfoStruct *)(arg))->UserData);
  threadInfo->mutex.Lock();
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  itk::ThreeDCircularProjectionGeometry::Pointer geometry = itk::ThreeDCircularProjectionGeometry::New();
  std::vector< std::string > fileNames;

  // Projections reader
  typedef itk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();

  // Create reconstructed image
  typedef itk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkinlinefdk>(constantImageSource, *(threadInfo->args_info));

  // Extract filter to process one projection at a time
  typedef itk::ExtractImageFilter<OutputImageType, OutputImageType> ExtractFilterType;
  ExtractFilterType::Pointer extract = ExtractFilterType::New();
  extract->SetInput( reader->GetOutput() );
  ExtractFilterType::InputImageRegionType subsetRegion;

  // Displaced detector weighting
  typedef itk::DisplacedDetectorImageFilter< OutputImageType > DDFType;
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput( extract->GetOutput() );
  ddf->SetGeometry( geometry );

  // Short scan image filter
//  typedef itk::ParkerShortScanImageFilter< OutputImageType > PSSFType;
//  PSSFType::Pointer pssf = PSSFType::New();
//  pssf->SetInput( ddf->GetOutput() );
//  pssf->SetGeometry( geometryReader->GetOutputObject() );
//  pssf->InPlaceOff();

  // This macro sets options for fdk filter which I can not see how to do better
  // because TFFTPrecision is not the same, e.g. for CPU and CUDA (SR)
#define SET_FELDKAMP_OPTIONS(f) \
    f->SetInput( 0, constantImageSource->GetOutput() ); \
    f->SetInput( 1, ddf->GetOutput() ); \
    f->SetGeometry( geometry ); \
    f->GetRampFilter()->SetTruncationCorrection(threadInfo->args_info->pad_arg); \
    f->GetRampFilter()->SetHannCutFrequency(threadInfo->args_info->hann_arg);

  // FDK reconstruction filtering
  itk::ImageToImageFilter<OutputImageType, OutputImageType>::Pointer feldkamp;
  typedef itk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
#if CUDA_FOUND
  typedef itk::CudaFDKConeBeamReconstructionFilter                FDKCUDAType;
#endif
#if OPENCL_FOUND
  typedef itk::OpenCLFDKConeBeamReconstructionFilter              FDKOPENCLType;
#endif
  if(!strcmp(threadInfo->args_info->hardware_arg, "cpu") )
    {
    feldkamp = FDKCPUType::New();
    SET_FELDKAMP_OPTIONS( static_cast<FDKCPUType*>(feldkamp.GetPointer()) );
    }
  else if(!strcmp(threadInfo->args_info->hardware_arg, "cuda") )
    {
#if CUDA_FOUND
    feldkamp = FDKCUDAType::New();
    SET_FELDKAMP_OPTIONS( static_cast<FDKCUDAType*>(feldkamp.GetPointer()) );
#else
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    exit(EXIT_FAILURE);
#endif
    }
  else if(!strcmp(threadInfo->args_info->hardware_arg, "opencl") )
    {
#if OPENCL_FOUND
    feldkamp = FDKOPENCLType::New();
    SET_FELDKAMP_OPTIONS( static_cast<FDKOPENCLType*>(feldkamp.GetPointer()) );
#else
    std::cerr << "The program has not been compiled with opencl option" << std::endl;
    exit(EXIT_FAILURE);
#endif
    }

  // Writer
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( threadInfo->args_info->output_arg );

  threadInfo->mutex.Unlock();

  // Inline loop
  std::cout << "Reconstruction thread has entered in the processing loop" << std::endl;
  for(;;)
  {
      threadInfo->mutex.Lock();

      if(geometry->GetMatrices().size()<threadInfo->nproj)
      {
          if(threadInfo->args_info->verbose_flag)
              std::cerr << "InlineThreadCallback has received projection #" << threadInfo->nproj-1 << std::endl;

          if(threadInfo->fileName != "" && (fileNames.size()==0 || fileNames.back() != threadInfo->fileName))
              fileNames.push_back(threadInfo->fileName);

          geometry->AddProjection(threadInfo->sid, threadInfo->sdd, threadInfo->gantryAngle,
                                  threadInfo->projOffsetX, threadInfo->projOffsetY,
                                  threadInfo->outOfPlaneAngle, threadInfo->inPlaneAngle,
                                  threadInfo->sourceOffsetX, threadInfo->sourceOffsetY);

          if(geometry->GetMatrices().size()!=threadInfo->nproj)
          {
              std::cerr << "Missed one projection in InlineThreadCallback" << std::endl;
              exit(EXIT_FAILURE);
          }
          if(geometry->GetMatrices().size()<3)
          {
              threadInfo->mutex.Unlock();
              continue;
          }

          reader->SetFileNames( fileNames );
          reader->UpdateOutputInformation();
          subsetRegion = reader->GetOutput()->GetLargestPossibleRegion();
          subsetRegion.SetIndex(Dimension-1, geometry->GetMatrices().size()-2);
          subsetRegion.SetSize(Dimension-1, 1);
          extract->SetExtractionRegion(subsetRegion);

          ddf->SetOffsets(threadInfo->minimumOffsetX, threadInfo->maximumOffsetX);

          TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );

          if(threadInfo->args_info->verbose_flag)
              std::cout << "Projection #" << subsetRegion.GetIndex(Dimension-1)
                        << " has been processed in reconstruction." << std::endl;

          OutputImageType::Pointer pimg = feldkamp->GetOutput();
          pimg->DisconnectPipeline();
          feldkamp->SetInput( pimg );
          TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->GetOutput()->UpdateOutputInformation() );
          TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->GetOutput()->PropagateRequestedRegion() );

          if(threadInfo->stop)
          {
              // Process first projection
              subsetRegion.SetIndex(Dimension-1, 0);
              extract->SetExtractionRegion(subsetRegion);
              TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
              if(threadInfo->args_info->verbose_flag)
                  std::cout << "Projection #" << subsetRegion.GetIndex(Dimension-1)
                            << " has been processed in reconstruction." << std::endl;
              pimg = feldkamp->GetOutput();
              pimg->DisconnectPipeline();
              feldkamp->SetInput( pimg );
              TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->GetOutput()->UpdateOutputInformation() );
              TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->GetOutput()->PropagateRequestedRegion() );

              // Process last projection
              subsetRegion.SetIndex(Dimension-1, geometry->GetMatrices().size()-1);
              extract->SetExtractionRegion(subsetRegion);
              TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() );
              if(threadInfo->args_info->verbose_flag)
                  std::cout << "Projection #" << subsetRegion.GetIndex(Dimension-1)
                            << " has been processed in reconstruction." << std::endl;

              //Write to disk and exit
              writer->SetInput( feldkamp->GetOutput() );
              TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );
              exit(EXIT_SUCCESS);
          }
      }

    threadInfo->mutex.Unlock();
    }
  return ITK_THREAD_RETURN_VALUE;
}

void computeOffsetsFromGeometry(itk::ThreeDCircularProjectionGeometry::Pointer geometry, double *minOffset, double *maxOffset)
{
    double min = itk::NumericTraits<double>::max();
    double max = itk::NumericTraits<double>::min();
    for(unsigned int i=0; i<geometry->GetProjectionOffsetsX().size(); i++)
      {
      min = vnl_math_min(min, geometry->GetProjectionOffsetsX()[i]);
      max = vnl_math_max(max, geometry->GetProjectionOffsetsX()[i]);
      }
    *minOffset = min;
    *maxOffset = max;
}





