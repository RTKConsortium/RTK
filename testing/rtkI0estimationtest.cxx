#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkI0EstimationProjectionFilter.h"
#include <itkRandomImageSource.h>

/**
 * \file rtkI0estimationtest.cxx
 *
 * \brief Test rtk::I0EstimationProjectionFilter
 *
 * \author Sebastien Brousmiche
 */

int main(int, char** )
{
  const unsigned int Dimension = 3;
	typedef itk::Image<unsigned short, Dimension> ImageType;
	
	typedef rtk::I0EstimationProjectionFilter<4> I0FilterType;
	I0FilterType::Pointer i0est = I0FilterType::New();
	
  // Constant image sources
  ImageType::SizeType size;
	size[0] = 128;
	size[1] = 200;
	size[2] = 1;
	ImageType::IndexType start;
	start.Fill(0);
	ImageType::RegionType region;
	region.SetIndex(start);
	region.SetSize(size);
		
	ImageType::Pointer projSource = ImageType::New();
	projSource->SetRegions(region);
	projSource->Allocate();
	projSource->FillBuffer(416);

	typedef itk::RandomImageSource< ImageType > RandomImageSourceType;
	RandomImageSourceType::Pointer randomSource = RandomImageSourceType::New();
	randomSource->SetMin(320);
	randomSource->SetMax(380);
	randomSource->SetSize(size);
	randomSource->Update();

	i0est->SetInput(randomSource->GetOutput());
	TRY_AND_EXIT_ON_ITK_EXCEPTION(i0est->UpdateOutputInformation());

	for (unsigned int i = 0; i < 10; ++i)  {

		//i0est->SetInput(projSource);
		i0est->SetInput(randomSource->GetOutput());

		i0est->Update();

		I0FilterType::HistogramType::Pointer hist;
		hist = i0est->GetOutput();

		//std::cout << hist << std::endl;
	}

	/*typedef itk::ImageRegionIterator< I0FilterType::HistogramType > IteratorType;
	IteratorType it(hist, hist->GetLargestPossibleRegion());
	it.GoToBegin();
	for (int i = 0; i < 20; ++i, ++it) 
		std::cout << it.Get() << " ";
	std::cout << std::endl;*/

  // If all succeed
	  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
