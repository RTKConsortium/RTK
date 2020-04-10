#include "rtkcheckimagequality_ggo.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionConstIterator.h"

/**
 * \file rtkcheckimagequality.cxx
 *
 * \brief Checks that an image has a satisfactory MSE against a reference.
 *
 * \author Aurélien Coussat
 */

template <class TImage>
double
MSE(typename TImage::Pointer reference, typename TImage::Pointer reconstruction)
{
  using ImageIteratorType = itk::ImageRegionConstIterator<TImage>;
  ImageIteratorType itTest(reconstruction, reconstruction->GetBufferedRegion());
  ImageIteratorType itRef(reference, reference->GetBufferedRegion());

  using ErrorType = double;
  ErrorType EnerError = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while (!itRef.IsAtEnd())
  {
    typename TImage::PixelType TestVal = itTest.Get();
    typename TImage::PixelType RefVal = itRef.Get();
    EnerError += std::pow(ErrorType(RefVal - TestVal), 2.);
    ++itTest;
    ++itRef;
  }

  return EnerError;
}

int
main(int argc, char ** argv)
{
  GGO(rtkcheckimagequality, args_info);

  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, Dimension>;

  using ReaderType = itk::ImageFileReader<ImageType>;
  ReaderType::Pointer reader;

  // Maximum number of comparisons to perform (depends on the number of inputs)
  unsigned int n_max =
    std::max({ args_info.reference_given, args_info.reconstruction_given, args_info.threshold_given });

  for (unsigned int i = 0; i < n_max; i++)
  {
    unsigned int reference_index = std::min(args_info.reference_given - 1, i);
    unsigned int reconstruction_index = std::min(args_info.reconstruction_given - 1, i);
    unsigned int threshold_index = std::min(args_info.threshold_given - 1, i);

    reader = ReaderType::New();
    reader->SetFileName(args_info.reference_arg[reference_index]);

    try
    {
      reader->Update();
    }
    catch (::itk::ExceptionObject & e)
    {
      std::cerr << e.GetDescription();
      return EXIT_FAILURE;
    }

    ImageType::Pointer reference = reader->GetOutput();

    reader = ReaderType::New();
    reader->SetFileName(args_info.reconstruction_arg[reconstruction_index]);

    try
    {
      reader->Update();
    }
    catch (::itk::ExceptionObject & e)
    {
      std::cerr << e.GetDescription();
      return EXIT_FAILURE;
    }

    ImageType::Pointer reconstruction = reader->GetOutput();

    double mse = MSE<ImageType>(reference, reconstruction);

    if (mse > args_info.threshold_arg[threshold_index])
    {
      std::cerr << "Error comparing " << args_info.reference_arg[reference_index] << " and "
                << args_info.reconstruction_arg[reconstruction_index] << ":" << std::endl
                << "MSE " << mse << " above given threshold " << args_info.threshold_arg[threshold_index] << std::endl;
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
