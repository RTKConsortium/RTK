/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkReconstructionConjugateGradientOperator.h"

namespace rtk
{

template<>
ReconstructionConjugateGradientOperator< itk::VectorImage<float, 3>, itk::Image<float, 3> >
::ReconstructionConjugateGradientOperator():
m_Geometry(ITK_NULLPTR),
m_Gamma(0),
m_Tikhonov(0)
{
this->SetNumberOfRequiredInputs(3);

// Create filters
#ifdef RTK_USE_CUDA
m_ConstantProjectionsSource = rtk::CudaConstantVolumeSource::New();
m_ConstantVolumeSource = rtk::CudaConstantVolumeSource::New();
m_LaplacianFilter = rtk::CudaLaplacianImageFilter::New();
#else
m_ConstantProjectionsSource = ConstantSourceType::New();
m_ConstantVolumeSource = ConstantSourceType::New();
//m_LaplacianFilter = LaplacianFilterType::New();
#endif
m_MatrixVectorMultiplyFilter = MatrixVectorMultiplyFilterType::New();
//m_MultiplyOutputVolumeFilter = MultiplyFilterType::New();
//m_MultiplyInputVolumeFilter = MultiplyFilterType::New();
m_MultiplySupportMaskFilter = MultiplyFilterType::New();
m_MultiplyTikhonovFilter = MultiplyFilterType::New();
m_AddTikhonovFilter = AddFilterType::New();

//m_MultiplyLaplacianFilter = MultiplyFilterType::New();

// Set permanent parameters
m_ConstantProjectionsSource->SetConstant(0.);
m_ConstantVolumeSource->SetConstant(0.);

// Set memory management options
m_ConstantProjectionsSource->ReleaseDataFlagOn();
m_ConstantVolumeSource->ReleaseDataFlagOn();
//m_LaplacianFilter->ReleaseDataFlagOn();
//m_MultiplyLaplacianFilter->ReleaseDataFlagOn();
}

template<>
void
ReconstructionConjugateGradientOperator< itk::VectorImage<float, 3>, itk::Image<float, 3> >
::GenerateOutputInformation()
{
  // Set runtime connections, and connections with
  // forward and back projection filters, which are set
  // at runtime
  m_ConstantVolumeSource->SetInformationFromImage(this->GetInput(0));
  m_ConstantProjectionsSource->SetInformationFromImage(this->GetInput(1));

  m_FloatingInputPointer = const_cast< itk::VectorImage<float, 3> *>(this->GetInput(0));

  // Set the first multiply filter to use the Support Mask, if any
  if (this->GetSupportMask().IsNotNull())
    {
    m_MultiplyInputVolumeFilter->SetInput1( m_FloatingInputPointer );
    m_MultiplyInputVolumeFilter->SetInput2( this->GetSupportMask() );
    m_FloatingInputPointer = m_MultiplyInputVolumeFilter->GetOutput();
    }

  // Set the forward projection filter's inputs
  m_ForwardProjectionFilter->SetInput(0, m_ConstantProjectionsSource->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, m_FloatingInputPointer);

  // Set the matrix vector multiply filter's inputs for multiplication
  // by the inverse covariance matrix (for GLS minimization)
  m_MatrixVectorMultiplyFilter->SetInput1(m_ForwardProjectionFilter->GetOutput()); // First input is the vector
  m_MatrixVectorMultiplyFilter->SetInput2(this->GetInput(2)); // Second input is the matrix

  // Set the back projection filter's inputs
  m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_MatrixVectorMultiplyFilter->GetOutput());
  m_FloatingOutputPointer= m_BackProjectionFilter->GetOutput();

  // Set the filters to compute the Tikhonov regularization, if any
  if (m_Tikhonov != 0)
    {
//    m_LaplacianFilter->SetInput(m_FloatingInputPointer);
//    m_MultiplyLaplacianFilter->SetInput1(m_LaplacianFilter->GetOutput());
    // Set "-1.0*gamma" because we need to perform "-1.0*Laplacian"
    // for correctly applying quadratic regularization || grad f ||_2^2
//    m_MultiplyLaplacianFilter->SetConstant2(-1.0*m_Gamma);
    m_MultiplyTikhonovFilter->SetInput(m_FloatingInputPointer);
    m_MultiplyTikhonovFilter->SetConstant2(m_Tikhonov);

    m_AddTikhonovFilter->SetInput(0, m_BackProjectionFilter->GetOutput());
//    m_AddFilter->SetInput(1, m_MultiplyLaplacianFilter->GetOutput());
    m_AddTikhonovFilter->SetInput(1, m_MultiplyTikhonovFilter->GetOutput());

    m_FloatingOutputPointer= m_AddTikhonovFilter->GetOutput();
    }

  // Set the second multiply filter to use the Support Mask, if any
  if (this->GetSupportMask().IsNotNull())
    {
    m_MultiplyOutputVolumeFilter->SetInput1( m_FloatingOutputPointer);
    m_MultiplyOutputVolumeFilter->SetInput2( this->GetSupportMask() );
    m_FloatingOutputPointer= m_MultiplyOutputVolumeFilter->GetOutput();
    }

  // Set geometry
  m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());

  // Set memory management parameters for forward
  // and back projection filters
  m_ForwardProjectionFilter->SetInPlace(true);
  m_ForwardProjectionFilter->ReleaseDataFlagOn();
  m_BackProjectionFilter->SetInPlace(true);
  m_BackProjectionFilter->SetReleaseDataFlag(this->GetSupportMask().IsNotNull() || (m_Tikhonov != 0));

  // Update output information on the last filter of the pipeline
  m_FloatingOutputPointer->UpdateOutputInformation();
  this->GetOutput()->CopyInformation( m_FloatingOutputPointer);
}

} // end namespace rtk
