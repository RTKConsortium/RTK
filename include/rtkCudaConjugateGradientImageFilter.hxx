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

#ifndef rtkCudaConjugateGradientImageFilter_hxx
#define rtkCudaConjugateGradientImageFilter_hxx

#include "rtkConfiguration.h"
//Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#include "rtkCudaConjugateGradientImageFilter.h"
#include "rtkCudaConjugateGradientImageFilter.hcu"
#include "rtkCudaConstantVolumeSource.h"

#include <itkMacro.h>

namespace rtk
{

template <class TImage>
CudaConjugateGradientImageFilter<TImage>
::CudaConjugateGradientImageFilter()
{
}

template <class TImage>
void
CudaConjugateGradientImageFilter<TImage>
::GPUGenerateData()
{
  typedef typename itk::PixelTraits<typename TImage::PixelType>::ValueType DataType;
  long int numberOfElements = this->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() * itk::PixelTraits<typename TImage::PixelType>::Dimension;

  // Create and allocate images
  typename TImage::Pointer P_k = TImage::New();
  typename TImage::Pointer R_k = TImage::New();
  P_k->SetRegions(this->GetOutput()->GetLargestPossibleRegion());
  R_k->SetRegions(this->GetOutput()->GetLargestPossibleRegion());
  P_k->Allocate();
  R_k->Allocate();
  P_k->CopyInformation(this->GetOutput());
  R_k->CopyInformation(this->GetOutput());

  // Copy the input to the output (X_0 = input)
  DataType *pin = *(DataType**)( this->GetX()->GetCudaDataManager()->GetGPUBufferPointer() );
  DataType *pX = *(DataType**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

  // On GPU, initialize the output to the input
  CUDA_copy(numberOfElements, pin, pX);

  // Compute A * X0
  this->m_A->SetX(this->GetX());
  this->m_A->Update();

  // Initialize AOut
  typename TImage::Pointer AOut = this->m_A->GetOutput();
  AOut->DisconnectPipeline();

  DataType *pR = *(DataType**)( R_k->GetCudaDataManager()->GetGPUBufferPointer() );
  DataType *pB = *(DataType**)( this->GetB()->GetCudaDataManager()->GetGPUBufferPointer() );
  DataType *pAOut = *(DataType**)( AOut->GetCudaDataManager()->GetGPUBufferPointer() );

  // Compute, on GPU, R_zero = P_zero = this->GetB() - this->m_A->GetOutput()
  CUDA_copy(numberOfElements, pB, pR);
  CUDA_subtract(numberOfElements, pR, pAOut);

  // B is now useless, and the memory it takes on the GPU is critical.
  // Transfer it back to the CPU memory
  this->GetB()->GetCudaDataManager()->GetCPUBufferPointer();

  DataType *pP = *(DataType**)( P_k->GetCudaDataManager()->GetGPUBufferPointer() );

  // P0 = R0
  CUDA_copy(numberOfElements, pR, pP);

  // Start iterations
  for (int iter=0; iter<this->m_NumberOfIterations; iter++)
    {
    // Compute A * P_k
    this->m_A->SetX(P_k);
    this->m_A->Update();
    AOut = this->m_A->GetOutput();
    AOut->DisconnectPipeline();

    DataType *pX = *(DataType**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );
    DataType *pR = *(DataType**)( R_k->GetCudaDataManager()->GetGPUBufferPointer() );
    DataType *pP = *(DataType**)( P_k->GetCudaDataManager()->GetGPUBufferPointer() );
    DataType *pAOut = *(DataType**)( AOut->GetCudaDataManager()->GetGPUBufferPointer() );

    // Compute, on GPU, alpha_k (only on GPU), X_k+1 (output), R_k+1, beta_k (only on GPU), P_k+1
    // The inputs are replaced by their next iterate
    CUDA_conjugate_gradient(numberOfElements, pX, pR, pP, pAOut);

    P_k->Modified();
    }

  P_k->ReleaseData();
  R_k->ReleaseData();
  AOut->ReleaseData();
}
} // namespace rtk

#endif
#endif
