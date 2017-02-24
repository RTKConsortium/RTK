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
m_Preconditioned(false),
m_Regularized(false),
m_Gamma(0)
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
//m_AddFilter = AddFilterType::New();

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
    m_ForwardProjectionFilter->SetInput(0, m_ConstantProjectionsSource->GetOutput());
    m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
    m_ConstantVolumeSource->SetInformationFromImage(this->GetInput(0));
    m_ConstantProjectionsSource->SetInformationFromImage(this->GetInput(1));
    m_MultiplySupportMaskFilter->SetInput2(this->GetSupportMask());
    m_ForwardProjectionFilter->SetInput(1, ApplySupportMask(this->GetInput(0)));

//    if (m_Regularized)
//      {
//      m_LaplacianFilter->SetInput(ApplySupportMask(this->GetInput(0)));

//      m_MultiplyLaplacianFilter->SetInput1(m_LaplacianFilter->GetOutput());
//      // Set "-1.0*gamma" because we need to perform "-1.0*Laplacian"
//      // for correctly applying quadratic regularization || grad f ||_2^2
//      m_MultiplyLaplacianFilter->SetConstant2(-1.0*m_Gamma);

//      m_AddFilter->SetInput1( m_BackProjectionFilter->GetOutput());
//      m_AddFilter->SetInput2( m_MultiplyLaplacianFilter->GetOutput());
//      }

    // Multiply the projections
    m_MatrixVectorMultiplyFilter->SetInput1(this->GetInput(2)); // First input is the matrix
    m_MatrixVectorMultiplyFilter->SetInput2(this->GetInput(1));  // Second input is the vector
    m_BackProjectionFilter->SetInput(1, m_MatrixVectorMultiplyFilter->GetOutput());

//    if (m_Preconditioned)
//      {
//      // Multiply the input volume
//      m_MultiplyInputVolumeFilter->SetInput1( ApplySupportMask(this->GetInput(0)) );
//      m_MultiplyInputVolumeFilter->SetInput2( this->GetInput(3) );
//      m_ForwardProjectionFilter->SetInput(1, m_MultiplyInputVolumeFilter->GetOutput());

//      // Multiply the volume
//      m_MultiplyOutputVolumeFilter->SetInput1(ApplySupportMask(m_BackProjectionFilter->GetOutput()));
//      m_MultiplyOutputVolumeFilter->SetInput2(this->GetInput(3));

//      // If a regularization is added, it needs to be added to the output of the
//      // m_MultiplyOutputVolumeFilter, instead of that of the m_BackProjectionFilter
//      if (m_Regularized)
//        {
//        m_LaplacianFilter->SetInput(m_MultiplyInputVolumeFilter->GetOutput());
//        m_MultiplyOutputVolumeFilter->SetInput1( ApplySupportMask(m_AddFilter->GetOutput()));
//        }
//      }

    // Set geometry
    m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
    m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());

    // Set memory management parameters for forward
    // and back projection filters
    m_ForwardProjectionFilter->SetInPlace(!m_Preconditioned);
    m_ForwardProjectionFilter->ReleaseDataFlagOn();

//    // Update output information on the last filter of the pipeline
//    if (m_Preconditioned)
//      {
//      m_MultiplyOutputVolumeFilter->UpdateOutputInformation();
//      this->GetOutput()->CopyInformation( m_MultiplyOutputVolumeFilter->GetOutput() );
//      }
//    else
//      {
//      if (m_Regularized)
//        {
//        m_AddFilter->UpdateOutputInformation();
//        this->GetOutput()->CopyInformation( m_AddFilter->GetOutput() );
//        }
//      else
//        {
        m_BackProjectionFilter->UpdateOutputInformation();
        this->GetOutput()->CopyInformation( m_BackProjectionFilter->GetOutput() );
//        }
//      }
}

template<>
void
ReconstructionConjugateGradientOperator< itk::VectorImage<float, 3>, itk::Image<float, 3> >
::GenerateData()
{
    // Execute Pipeline
//    if (m_Preconditioned)
//      {
//      m_MultiplyOutputVolumeFilter->Update();
//      this->GraftOutput( m_MultiplyOutputVolumeFilter->GetOutput() );

//      }
//    else
//      {
//      if (m_Regularized)
//        {
//        m_AddFilter->Update();
//        this->GraftOutput( const_cast<  itk::VectorImage<float, 3>* >(ApplySupportMask(m_AddFilter->GetOutput())) );
//        }
//      else
//        {
        m_BackProjectionFilter->Update();
        this->GraftOutput( const_cast< itk::VectorImage<float, 3>* >(ApplySupportMask(m_BackProjectionFilter->GetOutput())) );
//        }
//      }
}

} // end namespace rtk
