/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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

#include <itkProgressAccumulator.h>
#include <itkPixelTraits.h>

namespace rtk
{

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  ConjugateGradientConeBeamReconstructionFilter()
  : m_IterationReporter(this, 0, 1) // report every iteration
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_NumberOfIterations = 3;

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
  this->SetInput("InputWeights", const_cast<TWeightsImage *>(weights));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  SetLocalRegularizationWeights(const TSingleComponentImage * weights)
{
  this->SetInput("LocalRegularizationWeights", const_cast<TSingleComponentImage *>(weights));
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
  return static_cast<const TWeightsImage *>(this->itk::ProcessObject::GetInput("InputWeights"));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TSingleComponentImage::ConstPointer
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::
  GetLocalRegularizationWeights()
{
  return static_cast<const TSingleComponentImage *>(this->itk::ProcessObject::GetInput("LocalRegularizationWeights"));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TSingleComponentImage::ConstPointer
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::GetSupportMask()
{
  return static_cast<const TSingleComponentImage *>(this->itk::ProcessObject::GetInput("SupportMask"));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ConjugateGradientConeBeamReconstructionFilter<TOutputImage, TSingleComponentImage, TWeightsImage>::VerifyPreconditions()
  const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
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

  // Input "InputWeights" is the weights map on projections, either user-defined or filled with ones (default)
  if (this->GetInputWeights().IsNotNull())
  {
    typename TWeightsImage::Pointer inputWeights = const_cast<TWeightsImage *>(this->GetInputWeights().GetPointer());
    if (!inputWeights)
      return;
    inputWeights->SetRequestedRegion(inputWeights->GetLargestPossibleRegion());
  }

  // Input LocalRegularizationWeights is the optional weights map on regularization
  if (this->GetLocalRegularizationWeights().IsNotNull())
  {
    typename TSingleComponentImage::Pointer localRegWeights =
      const_cast<TSingleComponentImage *>(this->GetLocalRegularizationWeights().GetPointer());
    if (!localRegWeights)
      return;
    localRegWeights->SetRequestedRegion(localRegWeights->GetLargestPossibleRegion());
  }

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

  // Set forward projection filter
  m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter(this->m_CurrentForwardProjectionConfiguration);
  // Pass the ForwardProjection filter to the conjugate gradient operator
  m_CGOperator->SetForwardProjectionFilter(m_ForwardProjectionFilter);

  // Set back projection filter
  m_BackProjectionFilter = this->InstantiateBackProjectionFilter(this->m_CurrentBackProjectionConfiguration);
  // Pass the backprojection filter to the conjugate gradient operator and to the back projection filter generating the
  // B of AX=B
  m_BackProjectionFilterForB = this->InstantiateBackProjectionFilter(this->m_CurrentBackProjectionConfiguration);
  m_CGOperator->SetBackProjectionFilter(m_BackProjectionFilter);

  // Set runtime connections
  m_ConstantVolumeSource->SetInformationFromImage(this->GetInputVolume());
  m_CGOperator->SetInputProjectionStack(this->GetInputProjectionStack());
  m_CGOperator->SetSupportMask(this->GetSupportMask());
  m_ConjugateGradientFilter->SetX(this->GetInputVolume());
  m_DisplacedDetectorFilter->SetDisable(m_DisableDisplacedDetectorFilter);
  if (this->GetInputWeights().IsNull())
  {
    using PixelType = typename TWeightsImage::PixelType;
    using ComponentType = typename itk::PixelTraits<PixelType>::ValueType;
    auto ones = ConstantWeightSourceType::New();
    ones->SetInformationFromImage(this->GetInputProjectionStack());
    ones->SetConstant(PixelType(itk::NumericTraits<ComponentType>::One));
    ones->Update();
    this->SetInputWeights(ones->GetOutput());
  }
  m_DisplacedDetectorFilter->SetInput(this->GetInputWeights());

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_BackProjectionFilterForB->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_ConjugateGradientFilter->SetB(m_BackProjectionFilterForB->GetOutput());

  // Multiply the projections by the weights map
  m_MultiplyWithWeightsFilter->SetInput1(this->GetInputProjectionStack());
  m_MultiplyWithWeightsFilter->SetInput2(m_DisplacedDetectorFilter->GetOutput());
  m_CGOperator->SetInputWeights(m_DisplacedDetectorFilter->GetOutput());
  m_CGOperator->SetLocalRegularizationWeights(this->GetLocalRegularizationWeights());
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
  auto callbackCommand = itk::MemberCommand<Self>::New();
  callbackCommand->SetCallbackFunction(this, &Self::ReportProgress);
  m_ConjugateGradientFilter->AddObserver(itk::IterationEvent(), callbackCommand);

  m_ConjugateGradientFilter->Update();

  if (this->GetSupportMask())
  {
    m_MultiplyOutputFilter->Update();
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
