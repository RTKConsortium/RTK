/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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
#ifndef __itkCudaUnaryFunctorImageFilter_hxx
#define __itkCudaUnaryFunctorImageFilter_hxx

#include "itkCudaUnaryFunctorImageFilter.h"

namespace itk
{

template< class TInputImage, class TOutputImage, class TFunction, class TParentImageFilter >
void
CudaUnaryFunctorImageFilter< TInputImage, TOutputImage, TFunction, TParentImageFilter >
::GenerateOutputInformation()
{
  CPUSuperclass::GenerateOutputInformation();
}

template< class TInputImage, class TOutputImage, class TFunction, class TParentImageFilter >
void
CudaUnaryFunctorImageFilter< TInputImage, TOutputImage, TFunction, TParentImageFilter >
::GPUGenerateData()
{
    // Applying functor using GPU kernel
  typedef typename itk::CudaTraits< TInputImage >::Type  GPUInputImage;
  typedef typename itk::CudaTraits< TOutputImage >::Type GPUOutputImage;

  typename GPUInputImage::Pointer  inPtr =  dynamic_cast< GPUInputImage * >( this->ProcessObject::GetInput(0) );
  typename GPUOutputImage::Pointer otPtr =  dynamic_cast< GPUOutputImage * >( this->ProcessObject::GetOutput(0) );

  typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();

  int imgSize[3] = { 1, 1, 1 };

  int ImageDim = (int)TInputImage::ImageDimension;

  for (int i = 0; i < ImageDim; i++)
    {
    imgSize[i] = outSize[i];
    }

  size_t localSize[3], globalSize[3];
  localSize[0] = localSize[1] = localSize[2] = CudaGetLocalBlockSize(ImageDim);
  for (int i = 0; i < ImageDim; i++)
    {
    // total # of threads
    globalSize[i] = localSize[i]*(unsigned int)ceil((float)outSize[i]/(float)localSize[i]);
    }

  // arguments set up using Functor
  int argidx = (this->GetFunctor()).SetCudaKernelArguments(this->m_CudaKernelManager,
                                                           m_UnaryFunctorImageFilterCudaKernelHandle);
  // arguments set up
  this->m_CudaKernelManager->SetKernelArgWithImage(m_UnaryFunctorImageFilterCudaKernelHandle, argidx++,
                                                  inPtr->GetCudaDataManager());
  this->m_CudaKernelManager->SetKernelArgWithImage(m_UnaryFunctorImageFilterCudaKernelHandle, argidx++,
                                                  otPtr->GetCudaDataManager());
  for (int i=0; i<(int)TInputImage::ImageDimension; i++)
    {
    this->m_CudaKernelManager->SetKernelArg(m_UnaryFunctorImageFilterCudaKernelHandle, argidx++, sizeof(int),
                                           &(imgSize[i]));
    }

  // launch kernel
  this->m_CudaKernelManager->LaunchKernel(m_UnaryFunctorImageFilterCudaKernelHandle, ImageDim, globalSize, localSize);
}

} // end of namespace itk

#endif
