#include "itkProjectionsReader.h"

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkTimeProbe.h>


int main(int argc, char * argv[])
{
  if( argc < 4 ) {
    std::cerr << "Read projections and write them stacked to a single 3D image file." << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " directory regularExp outputImageFile " << std::endl;
    return EXIT_FAILURE;
  }

  typedef double OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(argv[1]);
  names->SetNumericSort(false);
  names->SetRegularExpression(argv[2]);
  names->SetSubMatch(0);

  // Projections reader
  typedef itk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( argv[3] );
  writer->SetInput( reader->GetOutput() );
  writer->UpdateOutputInformation();
  writer->SetNumberOfStreamDivisions( 1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() / (1024*1024*4) );

  try {
    writer->Update();
  } catch( itk::ExceptionObject & err ) {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
