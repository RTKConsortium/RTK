#include "rtkCudaFDKConeBeamReconstructionFilter.h"

rtk::CudaFDKConeBeamReconstructionFilter
::CudaFDKConeBeamReconstructionFilter():
    m_ExplicitGPUMemoryManagementFlag(false)
{
  // Create each filter which are specific for cuda
  m_RampFilter = RampFilterType::New();
  m_BackProjectionFilter = BackProjectionFilterType::New();

  //Permanent internal connections
  m_RampFilter->SetInput( m_WeightFilter->GetOutput() );
  m_BackProjectionFilter->SetInput( 1, m_RampFilter->GetOutput() );

  // Default parameters
  m_BackProjectionFilter->InPlaceOn();
  m_BackProjectionFilter->SetTranspose(false);
}

void
rtk::CudaFDKConeBeamReconstructionFilter
::GenerateData()
{
  // Init GPU memory
  if(!m_ExplicitGPUMemoryManagementFlag)
    this->InitDevice();

  // Run reconstruction
  this->Superclass::GenerateData();

  // Transfer result to CPU image
  if(!m_ExplicitGPUMemoryManagementFlag)
    this->CleanUpDevice();
}

void
rtk::CudaFDKConeBeamReconstructionFilter
::InitDevice()
{
  BackProjectionFilterType* cudabp = dynamic_cast<BackProjectionFilterType*>( m_BackProjectionFilter.GetPointer() );
  cudabp->InitDevice();
}

void
rtk::CudaFDKConeBeamReconstructionFilter
::CleanUpDevice()
{
  BackProjectionFilterType* cudabp = dynamic_cast<BackProjectionFilterType*>( m_BackProjectionFilter.GetPointer() );
  cudabp->CleanUpDevice();
}
