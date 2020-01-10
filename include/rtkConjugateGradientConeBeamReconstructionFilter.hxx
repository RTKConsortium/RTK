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

#ifndef rtkConjugateGradientConeBeamReconstructionFilter_hxx
#define rtkConjugateGradientConeBeamReconstructionFilter_hxx

#include "rtkConjugateGradientConeBeamReconstructionFilter.h"
#include <itkProgressAccumulator.h>

namespace rtk
{

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  ConjugateGradientConeBeamReconstructionFilter()
  : m_IterationReporter(this, 0, 1) // report every iteration
{
  this->SetNumberOfRequiredInputs(3);

  // Set the default values of member parameters
  m_NumberOfIterations = 3;
  m_IterationCosts = false;

  m_Gamma = 0;
  m_Tikhonov = 0;
  m_CudaConjugateGradient = true;
  m_DisableDisplacedDetectorFilter = false;

  // Create the filters
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();
  m_ConstantVolumeSource = ConstantImageSourceType::New();
  m_CGOperator = CGOperatorFilterType::New();
  m_MultiplyVolumeFilter = MultiplyFilterType::New();
  m_MultiplyProjectionsFilter = MultiplyFilterType::New();
  m_MultiplyOutputFilter = MultiplyFilterType::New();
  m_MultiplyWithWeightsFilter = MultiplyWithWeightsFilterType::New();

  // Set permanent parameters
  m_ConstantVolumeSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::SetInputVolume(
  const TOutputImage * vol)
{
  this->SetNthInput(0, const_cast<TOutputImage *>(vol));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  SetInputProjectionStack(const TOutputImage * projs)
{
  this->SetNthInput(1, const_cast<TOutputImage *>(projs));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::SetInputWeights(
  const TWeightsImage * weights)
{
  this->SetNthInput(2, const_cast<TWeightsImage *>(weights));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::SetSupportMask(
  const TSingleComponentImage * SupportMask)
{
  this->SetInput("SupportMask", const_cast<TSingleComponentImage *>(SupportMask));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TOutputImage::ConstPointer
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::GetInputVolume()
{
  return static_cast<const TOutputImage *>(this->itk::ProcessObject::GetInput(0));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TOutputImage::ConstPointer
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  GetInputProjectionStack()
{
  return static_cast<const TOutputImage *>(this->itk::ProcessObject::GetInput(1));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TWeightsImage::ConstPointer
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::GetInputWeights()
{
  return static_cast<const TWeightsImage *>(this->itk::ProcessObject::GetInput(2));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TSingleComponentImage::ConstPointer
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::GetSupportMask()
{
  return static_cast<const TSingleComponentImage *>(this->itk::ProcessObject::GetInput("SupportMask"));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
const std::vector<double> &
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::GetResidualCosts()
{
  return m_ConjugateGradientFilter->GetResidualCosts();
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  SetForwardProjectionFilter(ForwardProjectionType _arg)
{
  if (_arg != this->GetForwardProjectionFilter())
  {
    Superclass::SetForwardProjectionFilter(_arg);
    m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter(_arg);
    m_CGOperator->SetForwardProjectionFilter(m_ForwardProjectionFilter);
  }
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  SetBackProjectionFilter(BackProjectionType _arg)
{
  if (_arg != this->GetBackProjectionFilter())
  {
    Superclass::SetBackProjectionFilter(_arg);
    m_BackProjectionFilter = this->InstantiateBackProjectionFilter(_arg);
    m_BackProjectionFilterForB = this->InstantiateBackProjectionFilter(_arg);
    m_CGOperator->SetBackProjectionFilter(m_BackProjectionFilter);
  }
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  GenerateInputRequestedRegion()
{
  // Input 0 is the volume we update
  typename TOutputImage::Pointer inputPtr0 = const_cast<TOutputImage *>(this->GetInputVolume().GetPointer());
  if (!inputPtr0)
    return;
  inputPtr0->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

  // Input 1 is the stack of projections to backproject
  typename TOutputImage::Pointer inputPtr1 = const_cast<TOutputImage *>(this->GetInputProjectionStack().GetPointer());
  if (!inputPtr1)
    return;
  inputPtr1->SetRequestedRegion(inputPtr1->GetLargestPossibleRegion());

  // Input 2 is the weights map on projections, either user-defined or filled with ones (default)
  typename TWeightsImage::Pointer inputPtr2 = const_cast<TWeightsImage *>(this->GetInputWeights().GetPointer());
  if (!inputPtr2)
    return;
  inputPtr2->SetRequestedRegion(inputPtr2->GetLargestPossibleRegion());

  // Input "SupportMask" is the support constraint mask on volume, if any
  if (this->GetSupportMask().IsNotNull())
  {
    typename TSingleComponentImage::Pointer inputSupportMaskPtr =
      const_cast<TSingleComponentImage *>(this->GetSupportMask().GetPointer());
    if (!inputSupportMaskPtr)
      return;
    inputSupportMaskPtr->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
  }
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  GenerateOutputInformation()
{
  // Choose between cuda or non-cuda conjugate gradient filter
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
#ifdef RTK_USE_CUDA
  if (m_CudaConjugateGradient)
    m_ConjugateGradientFilter = InstantiateCudaConjugateGradientImageFilter<TOutputImage>();
#endif
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());
  m_ConjugateGradientFilter->SetIterationCosts(m_IterationCosts);

  // Set runtime connections
  m_ConstantVolumeSource->SetInformationFromImage(this->GetInputVolume());
  m_CGOperator->SetInputProjectionStack(this->GetInputProjectionStack());
  m_CGOperator->SetSupportMask(this->GetSupportMask());
  m_ConjugateGradientFilter->SetX(this->GetInputVolume());
  m_DisplacedDetectorFilter->SetDisable(m_DisableDisplacedDetectorFilter);
  m_DisplacedDetectorFilter->SetInput(this->GetInputWeights());

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_BackProjectionFilterForB->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_ConjugateGradientFilter->SetB(m_BackProjectionFilterForB->GetOutput());

  // Multiply the projections by the weights map
  m_MultiplyWithWeightsFilter->SetInput1(this->GetInputProjectionStack());
  m_MultiplyWithWeightsFilter->SetInput2(m_DisplacedDetectorFilter->GetOutput());
  m_CGOperator->SetInputWeights(m_DisplacedDetectorFilter->GetOutput());
  m_BackProjectionFilterForB->SetInput(1, m_MultiplyWithWeightsFilter->GetOutput());

  // If a support mask is used, it serves as preconditioning weights
  if (this->GetSupportMask().IsNotNull())
  {
    // Multiply the volume by support mask, and pass it to the conjugate gradient operator
    m_MultiplyVolumeFilter->SetInput1(m_BackProjectionFilterForB->GetOutput());
    m_MultiplyVolumeFilter->SetInput2(this->GetSupportMask());
    m_CGOperator->SetSupportMask(this->GetSupportMask());
    m_ConjugateGradientFilter->SetB(m_MultiplyVolumeFilter->GetOutput());

    // Multiply the output by the support mask
    m_MultiplyOutputFilter->SetInput1(m_ConjugateGradientFilter->GetOutput());
    m_MultiplyOutputFilter->SetInput2(this->GetSupportMask());
  }

  // For the same reason, set geometry now
  m_CGOperator->SetGeometry(this->m_Geometry);
  m_BackProjectionFilterForB->SetGeometry(this->m_Geometry.GetPointer());
  m_DisplacedDetectorFilter->SetGeometry(this->m_Geometry);

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_NumberOfIterations);
  m_CGOperator->SetGamma(m_Gamma);
  m_CGOperator->SetTikhonov(m_Tikhonov);

  // Set memory management parameters
  m_MultiplyProjectionsFilter->ReleaseDataFlagOn();
  m_BackProjectionFilterForB->ReleaseDataFlagOn();
  if (this->GetSupportMask().IsNotNull())
  {
    m_MultiplyVolumeFilter->ReleaseDataFlagOn();
    m_MultiplyOutputFilter->ReleaseDataFlagOn();
  }

  // Have the last filter calculate its output information
  m_ConjugateGradientFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation(m_ConjugateGradientFilter->GetOutput());
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::GenerateData()
{
  typename itk::MemberCommand<Self>::Pointer callbackCommand = itk::MemberCommand<Self>::New();
  callbackCommand->SetCallbackFunction(this, &Self::ReportProgress);
  m_ConjugateGradientFilter->AddObserver(itk::IterationEvent(), callbackCommand);

  m_ConjugateGradientFilter->Update();

  if (this->GetSupportMask())
  {
    m_MultiplyOutputFilter->Update();
  }

  if (this->GetSupportMask())
  {
    this->GraftOutput(m_MultiplyOutputFilter->GetOutput());
  }
  else
  {
    this->GraftOutput(m_ConjugateGradientFilter->GetOutput());
  }
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
template <
  typename ImageType,
  typename IterativeConeBeamReconstructionFilter<TOutputImage>::template EnableCudaScalarAndVectorType<ImageType> *>
typename ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  ConjugateGradientFilterPointer
  ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
    InstantiateCudaConjugateGradientImageFilter()
{
  ConjugateGradientFilterPointer cg;
#ifdef RTK_USE_CUDA
  cg = CudaConjugateGradientImageFilter<TOutputImage>::New();
#endif
  return cg;
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
template <
  typename ImageType,
  typename IterativeConeBeamReconstructionFilter<TOutputImage>::template DisableCudaScalarAndVectorType<ImageType> *>
typename ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  ConjugateGradientFilterPointer
  ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
    InstantiateCudaConjugateGradientImageFilter()
{
  itkGenericExceptionMacro(
    << "CudaConjugateGradientImageFilter only available with 3D CudaImage of float or itk::Vector<float,3>.");
  return nullptr;
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::ReportProgress(
  itk::Object *            caller,
  const itk::EventObject & event)
{
  if (!itk::IterationEvent().CheckEvent(&event))
  {
    return;
  }
  auto * cgCaller = dynamic_cast<rtk::ConjugateGradientImageFilter<TOutputImage> *>(caller);
  if (cgCaller)
  {
    this->GraftOutput(cgCaller->GetOutput());
    m_IterationReporter.CompletedStep();
  }
}

} // end namespace rtk


#endif
