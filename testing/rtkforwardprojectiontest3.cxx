
#include "rtkTestConfiguration.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"

// ITK includes
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

void DisplayImage()
{
  const unsigned int Dimension = 3;
  typedef float OutputPixelType;

  typedef itk::Image< OutputPixelType, Dimension > ImageType;
  typedef itk::ImageRegionConstIterator<ImageType> ImageIteratorType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( "/home/kitware/Desktop/data/img01.mha" );
  reader->Update();
  ImageIteratorType itRef( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );

  float min = 10000.;
  float max = -10000.;
  ImageType::RegionType region1 = reader->GetOutput()->GetLargestPossibleRegion();
  int size[3];
  size[0] = region1.GetSize()[0];
  size[1] = region1.GetSize()[1];
  size[2] = region1.GetSize()[2];
  itk::ImageRegionIterator< ImageType > itTemp(reader->GetOutput(), region1);
  for (unsigned int i = 0; i < size[0]*size[1]*size[2]; i++)
    {
    float value = itTemp.Get();
    if (value > max) max = value;
    if (value < min) min = value;
    ++itTemp;
    }
  for (unsigned int i = 0; i < size[2]; i++)
    {
    typedef itk::Image< unsigned char, 2 > ImageType2;
    ImageType2::Pointer image = ImageType2::New();
    ImageType2::RegionType region2;
    ImageType2::SizeType size2;
    size2[0] = size[0];
    size2[1] = size[1];
    region2.SetSize(size2);
    image->SetRegions(region2);
    image->Allocate();

    itk::ImageRegionIterator< ImageType > itProj(reader->GetOutput(), region1);
    itk::ImageRegionIterator< ImageType2 > itImage(image, region2);
    for(unsigned int e = 0; e < size[0]*size[1]*i; e++)
      {
      ++itProj;
      }
    for(unsigned int e = 0; e < size[0]*size[1]; e++)
      {
      itImage.Set((unsigned char)((itProj.Get()-min)/(max-min)*255));
      ++itProj;
      ++itImage;
      }

    std::ostringstream stm ;
    stm << i ;
    typedef itk::ImageFileWriter<ImageType2> WriterType2;
    WriterType2::Pointer writer2 = WriterType2::New();
    writer2->SetFileName( std::string("/home/kitware/Desktop/data2/test") + stm.str() + std::string(".png") );
    writer2->SetInput( image );
    writer2->Update();
    }
}
    
void CompareImages()
{
  const unsigned int Dimension = 3;
  typedef float OutputPixelType;

  typedef itk::Image< OutputPixelType, Dimension > ImageType;
  typedef itk::ImageRegionConstIterator<ImageType> ImageIteratorType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  
  std::string ref1 = "/home/kitware/Desktop/data/img01.mha";
  std::string test1 = "/home/kitware/Desktop/data/img02.mha";

  
    ReaderType::Pointer readerRef = ReaderType::New();
    readerRef->SetFileName( ref1 );
    readerRef->Update();
    std::cout << readerRef->GetOutput()->GetBufferedRegion().GetNumberOfPixels() << std::endl;
    ImageIteratorType itRef( readerRef->GetOutput(), readerRef->GetOutput()->GetBufferedRegion() );
  
    ReaderType::Pointer readerTest = ReaderType::New();
    readerTest->SetFileName( test1 );
    readerTest->Update();
    std::cout << readerTest->GetOutput()->GetBufferedRegion().GetNumberOfPixels() << std::endl;
    ImageIteratorType itTest( readerTest->GetOutput(), readerTest->GetOutput()->GetBufferedRegion() );

    typedef double ErrorType;
    ErrorType Error = 0.;
    ErrorType RelError = 0.;
    ErrorType EnerError = 0.;
    ErrorType MaxError = 0.;
    ErrorType MaxRelError = 0.;
    ErrorType MeanValueRef = 0.;
    ErrorType MaxValueRef = -1000000.;
    ErrorType MinValueRef = 1000000.;
    ErrorType MeanValueTest = 0.;
    ErrorType MaxValueTest = -1000000.;
    ErrorType MinValueTest = 1000000.;
    

    itTest.GoToBegin();
    itRef.GoToBegin();

    while( !itRef.IsAtEnd() )
      {
      ImageType::PixelType TestVal = itTest.Get();
      ImageType::PixelType RefVal = itRef.Get();

      MeanValueRef += RefVal;
      if (MaxValueRef < RefVal)
        {
        MaxValueRef = RefVal;
        }
      if (MinValueRef > RefVal)
        {
        MinValueRef = RefVal;
        }

      MeanValueTest += TestVal;
      if (MaxValueTest < TestVal)
        {
        MaxValueTest = TestVal;
        }
      if (MinValueTest > TestVal)
        {
        MinValueTest = TestVal;
        }
      
      if( TestVal != RefVal )
        {
        Error += vcl_abs(RefVal - TestVal);
        EnerError += vcl_pow(ErrorType(RefVal - TestVal), 2.);
        if (MaxError < vcl_abs(RefVal - TestVal))
          {
          MaxError = vcl_abs(RefVal - TestVal);
          }
        if (RefVal != 0)
          {
          RelError += vcl_abs((RefVal - TestVal)/RefVal);
          if (MaxRelError < vcl_abs((RefVal - TestVal)/RefVal))
            {
            MaxRelError = vcl_abs((RefVal - TestVal)/RefVal);
            }
          }
        }
      ++itTest;
      ++itRef;
      }

    // Error per Pixel
    ErrorType ErrorPerPixel = Error/readerRef->GetOutput()->GetBufferedRegion().GetNumberOfPixels();
    std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
    // Error Max
    std::cout << "Error max = " << MaxError << std::endl;
    
    // Relative error per Pixel
    ErrorType RelErrorPerPixel = RelError/readerRef->GetOutput()->GetBufferedRegion().GetNumberOfPixels();
    std::cout << "Error relative per pixel = " << RelErrorPerPixel << std::endl;
    // Relative error Max
    std::cout << "Error relative max = " << MaxRelError << std::endl;
    
    // Value Mean
    MeanValueRef = MeanValueRef / readerRef->GetOutput()->GetBufferedRegion().GetNumberOfPixels();
    std::cout << "Mean value ref = " << MeanValueRef << std::endl;
    // Value Max
    std::cout << "Max value ref = " << MaxValueRef << std::endl;
    // Value Min
    std::cout << "Min value ref = " << MinValueRef << std::endl;
    
    // Value Mean
    MeanValueTest = MeanValueTest / readerRef->GetOutput()->GetBufferedRegion().GetNumberOfPixels();
    std::cout << "Mean value test = " << MeanValueTest << std::endl;
    // Value Max
    std::cout << "Max value test = " << MaxValueTest << std::endl;
    // Value Min
    std::cout << "Min value test = " << MinValueTest << std::endl;
    
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

int main(int , char** )
{
  DisplayImage();
  CompareImages();
  getchar();

  return EXIT_SUCCESS;
}
