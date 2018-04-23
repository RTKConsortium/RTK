#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkMinimumMaximumImageCalculator.h"

/**
 * \file rtkcheckimagequality.cxx
 *
 * \brief Check the quality of the input image
 * If a baseline is provided then a pixel quality 
 *
 * \author Julien Jomier
 */
int main(int argc, char** argv)
{
  const unsigned int Dimension = 2;
  typedef float                              PixelType;
  typedef itk::Image< PixelType, Dimension > ImageType;

  if(argc < 2)
    {
    std::cout << "Usage: " << argv[0] << " inputimage [baseline]" << std::endl;
    return EXIT_FAILURE;
    }

  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);
  
  try
    {
    reader->Update();
    }
  catch(::itk::ExceptionObject e)
    {
    std::cerr << e.GetDescription();
    return EXIT_FAILURE;
    }

  typedef itk::MinimumMaximumImageCalculator<ImageType> CalculatorType;
  CalculatorType::Pointer calculator = CalculatorType::New();
  calculator->SetImage(reader->GetOutput());
  calculator->Compute();

  float minimum = calculator->GetMinimum();
  float maximum = calculator->GetMaximum();

  if(maximum-minimum < 0.00001)
    {
    std::cout << "Minimum intensity = " << minimum << std::endl;
    std::cout << "Maximum intensity = " << minimum << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "Image Quality Valid" << std::endl;
  return EXIT_SUCCESS;
}
