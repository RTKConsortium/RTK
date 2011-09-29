namespace itk
{

OpenCLFDKConeBeamReconstructionFilter
::OpenCLFDKConeBeamReconstructionFilter()
{
  // Create each filter which are specific for OpenCL
  m_BackProjectionFilter = BackProjectionFilterType::New();

  //Permanent internal connections
  m_BackProjectionFilter->SetInput( 1, m_RampFilter->GetOutput() );

  // Default parameters
  m_BackProjectionFilter->InPlaceOn();
  m_BackProjectionFilter->SetTranspose(false);
}

void
OpenCLFDKConeBeamReconstructionFilter
::GenerateData()
{
  BackProjectionFilterType* openclbp = dynamic_cast<BackProjectionFilterType*>( m_BackProjectionFilter.GetPointer() );

  // Init GPU memory
  openclbp->InitDevice();

  // Run reconstruction
  this->Superclass::GenerateData();

  // Transfer result to CPU image
  openclbp->CleanUpDevice();
}

} // end namespace itk
