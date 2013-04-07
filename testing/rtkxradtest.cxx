#include "rtkTestConfiguration.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include "rtkXRadGeometryReader.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkRegularExpressionSeriesFileNames.h>

typedef rtk::ThreeDCircularProjectionGeometry GeometryType;

template<class TImage>
void CheckImageQuality(typename TImage::Pointer recon, typename TImage::Pointer ref)
{
#if !(FAST_TESTS_NO_CHECKS)
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
  if (ErrorPerPixel > 1.6e-7)
    {
    std::cerr << "Test Failed, Error per pixel not valid! "
              << ErrorPerPixel << " instead of 1.6e-7" << std::endl;
    exit( EXIT_FAILURE);
    }
  if (PSNR < 100.)
    {
    std::cerr << "Test Failed, PSNR not valid! "
              << PSNR << " instead of 100" << std::endl;
    exit( EXIT_FAILURE);
    }
#endif
}

void CheckGeometries(GeometryType *g1, GeometryType *g2)
{
#if !(FAST_TESTS_NO_CHECKS)
  const double e           = 1e-10;
  const unsigned int nproj = g1->GetGantryAngles().size();
  if(g2->GetGantryAngles().size() != nproj)
    { 
    std::cerr << "Unequal number of projections in the two geometries"
              << std::endl;
    exit(EXIT_FAILURE);
    }

  for(unsigned int i=0; i<nproj; i++)
    {
    if( e < std::fabs(g1->GetGantryAngles()[i] - 
                      g2->GetGantryAngles()[i]) ||
        e < std::fabs(g1->GetOutOfPlaneAngles()[i] -
                      g2->GetOutOfPlaneAngles()[i]) || 
        e < std::fabs(g1->GetInPlaneAngles()[i] - 
                      g2->GetInPlaneAngles()[i]) ||
        e < std::fabs(g1->GetSourceToIsocenterDistances()[i] -
                      g2->GetSourceToIsocenterDistances()[i]) || 
        e < std::fabs(g1->GetSourceOffsetsX()[i] - 
                      g2->GetSourceOffsetsX()[i]) || 
        e < std::fabs(g1->GetSourceOffsetsY()[i] -
                      g2->GetSourceOffsetsY()[i]) || 
        e < std::fabs(g1->GetSourceToDetectorDistances()[i] -
                      g2->GetSourceToDetectorDistances()[i]) || 
        e < std::fabs(g1->GetProjectionOffsetsX()[i] -
                      g2->GetProjectionOffsetsX()[i]) || 
        e < std::fabs(g1->GetProjectionOffsetsY()[i] -
                      g2->GetProjectionOffsetsY()[i]) )
      {
      std::cerr << "Geometry of projection #" << i << "is unvalid."
                << std::endl;
      exit(EXIT_FAILURE);
      }

    }
#endif
}

int main(int, char** )
{
  // Elekta geometry
  rtk::XRadGeometryReader::Pointer geoTargReader;
  geoTargReader = rtk::XRadGeometryReader::New();
  geoTargReader->SetImageFileName( std::string(RTK_DATA_ROOT) +
                                   std::string("/Input/XRad/SolidWater_HiGain1x1.header") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoTargReader->UpdateOutputData() );

  // Reference geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geoRefReader;
  geoRefReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geoRefReader->SetFilename( std::string(RTK_DATA_ROOT) +
                             std::string("/Baseline/XRad/geometry.xml") );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geoRefReader->GenerateOutputInformation() )

  // 1. Check geometries
  CheckGeometries(geoTargReader->GetGeometry(), geoRefReader->GetOutputObject() );

  // ******* COMPARING projections *******
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > ImageType;

  // Elekta projections reader
  typedef rtk::ProjectionsReader< ImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  std::vector<std::string> fileNames;
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Input/XRad/SolidWater_HiGain1x1_firstProj.header") );
  reader->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );

  // Reference projections reader
  ReaderType::Pointer readerRef = ReaderType::New();
  fileNames.clear();
  fileNames.push_back( std::string(RTK_DATA_ROOT) +
                       std::string("/Baseline/XRad/attenuation.mha") );
  readerRef->SetFileNames( fileNames );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(readerRef->Update());

  // 2. Compare read projections
  CheckImageQuality< ImageType >(reader->GetOutput(), readerRef->GetOutput());

  // If both succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
