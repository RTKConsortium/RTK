//#include "itkFFTWCommon.h"
//#include "itkFFTWComplexConjugateToRealImageFilter.h"
//#include "itkFFTWRealToComplexConjugateImageFilter.h"
#include "rtkAdditiveGaussianNoiseImageFilter.h"
#include "rtkAmsterdamShroudImageFilter.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkBoellaardScatterCorrectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkDisplacedDetectorImageFilter.h"
#include "rtkDrawGeometricPhantomImageFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkElektaSynergyLookupTableImageFilter.h"
#include "rtkFDKBackProjectionImageFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkFDKWeightProjectionFilter.h"
#include "rtkFFTRampImageFilter.h"
#include "rtkFieldOfViewImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkGeometricPhantomFileReader.h"
#include "rtkGgoFunctions.h"
#include "rtkHisImageIO.h"
#include "rtkHisImageIOFactory.h"
#include "rtkHndImageIO.h"
#include "rtkHndImageIOFactory.h"
#include "rtkHomogeneousMatrix.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkLookupTableImageFilter.h"
#include "rtkMacro.h"
#include "rtkParkerShortScanImageFilter.h"
#include "rtkProjectGeometricPhantomImageFilter.h"
#include "rtkProjectionGeometry.h"
#include "rtkProjectionsReader.h"
#include "rtkRayBoxIntersectionFunction.h"
#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkRayCastInterpolateImageFunction.h"
#include "rtkRayCastInterpolatorForwardProjectionImageFilter.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkRayQuadricIntersectionFunction.h"
#include "rtkRayQuadricIntersectionImageFilter.h"
#include "rtkSARTConeBeamReconstructionFilter.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkVarianObiRawImageFilter.h"
#include "rtkXRadGeometryReader.h"
#include "rtkGgoArgsInfoManager.h"

//#ifdef USE_OPENCL
//  #include "rtkOpenCLFDKBackProjectionImageFilter.h"
//  #include "rtkOpenCLFDKConeBeamReconstructionFilter.h"
//  #include "rtkOpenCLUtilities.h"
//#endif
#ifdef USE_CUDA
  #include "rtkCudaFDKBackProjectionImageFilter.h"
  #include "rtkCudaFDKConeBeamReconstructionFilter.h"
  #include "rtkCudaFFTRampImageFilter.h"
#endif

/**
 * \file rtkheadertest.cxx
 *
 * \brief This test includes all headers for coverage and style purposed.
 *
 * \author Simon Rit
 */

int main(int , char**)
{
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
