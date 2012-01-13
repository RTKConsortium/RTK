#ifndef __itkCudaFDKConeBeamReconstructionFilter_h
#define __itkCudaFDKConeBeamReconstructionFilter_h

#include "itkFDKConeBeamReconstructionFilter.h"
#include "itkCudaFFTRampImageFilter.h"
#include "itkCudaFDKBackProjectionImageFilter.h"

/** \class CudaFDKConeBeamReconstructionFilter
 * \brief Implements Feldkamp, David and Kress cone-beam reconstruction using Cuda
 *
 * Replaces ramp and backprojection in FDKConeBeamReconstructionFilter with
 * - CudaFFTRampImageFilter
 * - CudaFDKBackProjectionImageFilter.
 * Also take care to create the reconstructed volume on the GPU at the beginning and
 * transfers it at the end.
 *
 * \author Simon Rit
 */
namespace itk
{

class ITK_EXPORT CudaFDKConeBeamReconstructionFilter :
  public FDKConeBeamReconstructionFilter< itk::Image<float,3>, itk::Image<float,3>, float >
{
public:
  /** Standard class typedefs. */
  typedef CudaFDKConeBeamReconstructionFilter                                                Self;
  typedef FDKConeBeamReconstructionFilter< itk::Image<float,3>, itk::Image<float,3>, float > Superclass;
  typedef SmartPointer<Self>                                                                 Pointer;
  typedef SmartPointer<const Self>                                                           ConstPointer;

  /** Typedefs of subfilters which have been implemented with CUDA */
  typedef itk::CudaFFTRampImageFilter           RampFilterType;
  typedef itk::CudaFDKBackProjectionImageFilter BackProjectionFilterType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaFDKConeBeamReconstructionFilter, ImageToImageFilter);

protected:
  CudaFDKConeBeamReconstructionFilter();
  ~CudaFDKConeBeamReconstructionFilter(){}

  void GenerateData();

private:
  //purposely not implemented
  CudaFDKConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);
}; // end of class

} // end namespace itk

#endif
