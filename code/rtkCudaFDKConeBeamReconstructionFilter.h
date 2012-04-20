#ifndef __rtkCudaFDKConeBeamReconstructionFilter_h
#define __rtkCudaFDKConeBeamReconstructionFilter_h

#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkCudaFFTRampImageFilter.h"
#include "rtkCudaFDKBackProjectionImageFilter.h"

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
namespace rtk
{

class ITK_EXPORT CudaFDKConeBeamReconstructionFilter :
  public FDKConeBeamReconstructionFilter< itk::Image<float,3>, itk::Image<float,3>, float >
{
public:
  /** Standard class typedefs. */
  typedef CudaFDKConeBeamReconstructionFilter                                                Self;
  typedef FDKConeBeamReconstructionFilter< itk::Image<float,3>, itk::Image<float,3>, float > Superclass;
  typedef itk::SmartPointer<Self>                                                                 Pointer;
  typedef itk::SmartPointer<const Self>                                                           ConstPointer;

  /** Typedefs of subfilters which have been implemented with CUDA */
  typedef rtk::CudaFFTRampImageFilter           RampFilterType;
  typedef rtk::CudaFDKBackProjectionImageFilter BackProjectionFilterType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaFDKConeBeamReconstructionFilter, FDKConeBeamReconstructionFilter);

  /** Functions to init and clean up the GPU when ExplicitGPUMemoryManagementFlag is true. */
  void InitDevice();
  void CleanUpDevice();

  /** Boolean to keep the hand on the memory management of the GPU. Default is
   * off. If on, the user must call manually InitDevice and CleanUpDevice. */
  itkGetMacro(ExplicitGPUMemoryManagementFlag, bool);
  itkSetMacro(ExplicitGPUMemoryManagementFlag, bool);

protected:
  CudaFDKConeBeamReconstructionFilter();
  ~CudaFDKConeBeamReconstructionFilter(){}

  void GenerateData();

private:
  //purposely not implemented
  CudaFDKConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

  bool m_ExplicitGPUMemoryManagementFlag;
}; // end of class

} // end namespace rtk

#endif
