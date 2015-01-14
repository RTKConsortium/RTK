#include "rtkTest.h"
#include "rtkTestConfiguration.h"
#include "rtkMacro.h"
#include "rtkLagCorrectionImageFilter.h"

#include "itkRealTimeClock.h"

#include <vector>

using namespace std;

/**
 * \file rtklagcorrectiontest.cxx
 *
 * \brief 
 *
 * Description
 *
 * \author Sebastien Brousmiche
 */

//
// TODO: - why transfer to constant memory does not work?
//       - Access to parameters a and b?

const unsigned ModelOrder = 4;
const unsigned Nprojections = 10;

int main(int argc, char * argv[])
{
	const unsigned int Dimension = 3;
	typedef float PixelType;
	
	typedef itk::Image< PixelType, Dimension > ImageType;
	typedef itk::Vector<float, ModelOrder> VectorType;     // Parameter type always float/double

	typedef rtk::LagCorrectionImageFilter< ImageType, ModelOrder> LCImageFilterType;
	LCImageFilterType::Pointer lagcorr = LCImageFilterType::New();

	typedef itk::RealTimeClock itkClockType;
	typedef itk::RealTimeStamp::TimeRepresentationType itkTimeType;

	itkClockType::Pointer clock = itkClockType::New();

	ImageType::SizeType size;
	size[0] = 650;
	size[1] = 700;
	size[2] = 1;

	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;

	ImageType::RegionType region;
	region.SetSize(size);
	region.SetIndex(start);

	VectorType a;
	a[0] = 0.7055f;
	a[1] = 0.1141f;
	a[2] = 0.0212f;
	a[3] = 0.0033f;
	
	VectorType b;
	b[0] = 2.911e-3f;
	b[1] = 0.4454e-3f;
	b[2] = 0.0748e-3f;
	b[3] = 0.0042e-3f;

	lagcorr->SetA(a);
	lagcorr->SetB(b);
	lagcorr->Initialize();

	std::cout << "a = "<<lagcorr->GetA()<<std::endl;
	std::cout << "b = "<<lagcorr->GetB() << std::endl;
    
	for (unsigned i = 0; i < Nprojections; ++i) {
		std::cout << "Send image no " << i << std::endl;
		ImageType::Pointer inputI = ImageType::New();
		inputI->SetRegions(region);
		inputI->Allocate();
	  inputI->FillBuffer(1.0f);

		itkTimeType T1 = clock->GetRealTimeStamp().GetTimeInMicroSeconds();
		lagcorr->SetInput(inputI.GetPointer());

		TRY_AND_EXIT_ON_ITK_EXCEPTION( lagcorr->Update() )

		itkTimeType T2 = clock->GetRealTimeStamp().GetTimeInMicroSeconds();
		std::cout << (T2-T1) <<" us"<< std::endl;
	}

	//lagcorr->ResetInternalState();
	
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
