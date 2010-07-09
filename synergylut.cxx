#include "itkSynergyLutImageFilter.h"
#include "itkHisImageIOFactory.h"
#include <itkImageSeriesReader.h>
#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkTimeProbe.h>

//--------------------------------------------------------------------
int main(int argc, char * argv[])
{
  if( argc < 4 ) {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " directory regularExp outputImageFile " << std::endl;
    return EXIT_FAILURE;
  }

  typedef unsigned short InputPixelType;
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< InputPixelType, Dimension > InputImageType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(argv[1]);
  names->SetNumericSort(false);
  names->SetRegularExpression(argv[2]);
  names->SetSubMatch(0);

  // Read
  typedef itk::ImageSeriesReader< InputImageType >  ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetImageIO( itk::HisImageIO::New() );
  reader->SetFileNames( names->GetFileNames() );

  // Lut filter
  typedef itk::SynergyLutImageFilter<InputImageType, OutputImageType> lutFilterType;
  lutFilterType::Pointer lutFilter = lutFilterType::New();
  lutFilter->SetInput( reader->GetOutput() );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[3] );
  writer->SetInput( lutFilter->GetOutput() );
  writer->UpdateOutputInformation();
  writer->SetNumberOfStreamDivisions( 1+lutFilter->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() / (1024*1024*16) );

  try {
    writer->Update();
  } catch( itk::ExceptionObject & err ) {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
