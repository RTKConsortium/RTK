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

#ifndef rtkReconstructionConjugateGradientOperator_hxx
#define rtkReconstructionConjugateGradientOperator_hxx


namespace rtk
{

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::
  ReconstructionConjugateGradientOperator()
{
  this->SetNumberOfRequiredInputs(3);

  // Create filters
  // #ifdef RTK_USE_CUDA
  //  m_ConstantProjectionsSource = rtk::CudaConstantVolumeSource::New();
  //  m_ConstantVolumeSource = rtk::CudaConstantVolumeSource::New();
  //  m_LaplacianFilter = rtk::CudaLaplacianImageFilter::New();
  // #else
  m_ConstantProjectionsSource = ConstantSourceType::New();
  m_ConstantVolumeSource = ConstantSourceType::New();
  // #endif
  m_MultiplyWithWeightsFilter = MultiplyWithWeightsFilterType::New();
  m_MultiplyOutputVolumeFilter = MultiplyFilterType::New();
  m_MultiplyInputVolumeFilter = MultiplyFilterType::New();

  // Set permanent parameters
  m_ConstantProjectionsSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_ConstantVolumeSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());

  // Set memory management options
  m_ConstantProjectionsSource->ReleaseDataFlagOn();
  m_ConstantVolumeSource->ReleaseDataFlagOn();
  //  m_LaplacianFilter->ReleaseDataFlagOn();
  //  m_MultiplyLaplacianFilter->ReleaseDataFlagOn();
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::SetInputVolume(
  const TOutputImage * vol)
{
  this->SetNthInput(0, const_cast<TOutputImage *>(vol));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::SetInputProjectionStack(
  const TOutputImage * projs)
{
  this->SetNthInput(1, const_cast<TOutputImage *>(projs));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::SetInputWeights(
  const TWeightsImage * weights)
{
  this->SetNthInput(2, const_cast<TWeightsImage *>(weights));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::SetSupportMask(
  const TSingleComponentImage * SupportMask)
{
  this->SetInput("SupportMask", const_cast<TSingleComponentImage *>(SupportMask));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TOutputImage::ConstPointer
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::GetInputVolume()
{
  return static_cast<const TOutputImage *>(this->itk::ProcessObject::GetInput(0));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::
  SetLocalRegularizationWeights(const TSingleComponentImage * localRegularizationWeights)
{
  this->SetInput("LocalRegularizationWeights", const_cast<TSingleComponentImage *>(localRegularizationWeights));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TOutputImage::ConstPointer
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::GetInputProjectionStack()
{
  return static_cast<const TOutputImage *>(this->itk::ProcessObject::GetInput(1));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TWeightsImage::ConstPointer
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::GetInputWeights()
{
  return static_cast<const TWeightsImage *>(this->itk::ProcessObject::GetInput(2));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TSingleComponentImage::ConstPointer
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::GetSupportMask()
{
  return static_cast<const TSingleComponentImage *>(this->itk::ProcessObject::GetInput("SupportMask"));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
typename TSingleComponentImage::ConstPointer
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::
  GetLocalRegularizationWeights()
{
  return static_cast<const TSingleComponentImage *>(this->itk::ProcessObject::GetInput("LocalRegularizationWeights"));
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::SetBackProjectionFilter(
  BackProjectionFilterType * _arg)
{
  m_BackProjectionFilter = _arg;
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::SetForwardProjectionFilter(
  ForwardProjectionFilterType * _arg)
{
  m_ForwardProjectionFilter = _arg;
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::
  GenerateInputRequestedRegion()
{
  // Input 0 is the volume in which we backproject
  typename TOutputImage::Pointer inputPtr0 = const_cast<TOutputImage *>(this->GetInputVolume().GetPointer());
  if (!inputPtr0)
    return;
  inputPtr0->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

  // Input 1 is the stack of projections to backproject
  typename TOutputImage::Pointer inputPtr1 = const_cast<TOutputImage *>(this->GetInputProjectionStack().GetPointer());
  if (!inputPtr1)
    return;
  inputPtr1->SetRequestedRegion(inputPtr1->GetLargestPossibleRegion());

  // Input 2 is the weights map on projections, if any
  typename TWeightsImage::Pointer inputWeights = const_cast<TWeightsImage *>(this->GetInputWeights().GetPointer());
  if (!inputWeights)
    return;
  inputWeights->SetRequestedRegion(inputWeights->GetLargestPossibleRegion());

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
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::GenerateOutputInformation()
{
  // Set runtime connections, and connections with
  // forward and back projection filters, which are set
  // at runtime
  m_ConstantVolumeSource->SetInformationFromImage(this->GetInputVolume());
  m_ConstantProjectionsSource->SetInformationFromImage(this->GetInputProjectionStack());

  m_FloatingInputPointer = const_cast<TOutputImage *>(this->GetInputVolume().GetPointer());

  // Set the first multiply filter to use the Support Mask, if any
  if (this->GetSupportMask().IsNotNull())
  {
    m_MultiplyInputVolumeFilter->SetInput1(m_FloatingInputPointer);
    m_MultiplyInputVolumeFilter->SetInput2(this->GetSupportMask());
    m_FloatingInputPointer = m_MultiplyInputVolumeFilter->GetOutput();
  }

  // Set the forward projection filter's inputs
  m_ForwardProjectionFilter->SetInput(0, m_ConstantProjectionsSource->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, m_FloatingInputPointer);

  // Set the multiply filter's inputs for the projection weights (for WLS minimization)
  m_MultiplyWithWeightsFilter->SetInput1(m_ForwardProjectionFilter->GetOutput());
  m_MultiplyWithWeightsFilter->SetInput2(this->GetInputWeights());

  // Set the back projection filter's inputs
  m_BackProjectionFilter->SetInput(0, m_ConstantVolumeSource->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_MultiplyWithWeightsFilter->GetOutput());
  m_FloatingOutputPointer = m_BackProjectionFilter->GetOutput();

  // Set the filters to compute the laplacian regularization, if any
  if (m_Gamma != 0)
  {
    m_FloatingOutputPointer = ConnectGradientRegularization<TOutputImage>();
  }

  // Set the filters to compute the Tikhonov regularization, if any
  if (m_Tikhonov != 0)
  {
    m_AddTikhonovFilter = AddFilterType::New();
    m_AddTikhonovFilter->SetInput(1, m_FloatingOutputPointer);

    m_MultiplyTikhonovFilter = MultiplyFilterType::New();
    m_MultiplyTikhonovFilter->SetInput(m_FloatingInputPointer);
    m_MultiplyTikhonovFilter->SetConstant2(m_Tikhonov);
    m_FloatingOutputPointer = m_MultiplyTikhonovFilter->GetOutput();

    if (this->GetLocalRegularizationWeights().IsNotNull())
    {
      m_MultiplyTikhonovWeightsFilter = MultiplyFilterType::New();
      m_MultiplyTikhonovWeightsFilter->SetInput1(m_FloatingOutputPointer);
      m_MultiplyTikhonovWeightsFilter->SetInput2(this->GetLocalRegularizationWeights());
      m_FloatingOutputPointer = m_MultiplyTikhonovWeightsFilter->GetOutput();
    }

    m_AddTikhonovFilter->SetInput(0, m_FloatingOutputPointer);

    m_FloatingOutputPointer = m_AddTikhonovFilter->GetOutput();
  }

  // Set the second multiply filter to use the Support Mask, if any
  if (this->GetSupportMask().IsNotNull())
  {
    m_MultiplyOutputVolumeFilter->SetInput1(m_FloatingOutputPointer);
    m_MultiplyOutputVolumeFilter->SetInput2(this->GetSupportMask());
    m_FloatingOutputPointer = m_MultiplyOutputVolumeFilter->GetOutput();
  }

  // Set geometry
  m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());

  // Set memory management parameters for forward
  // and back projection filters
  m_ForwardProjectionFilter->SetInPlace(true);
  m_ForwardProjectionFilter->ReleaseDataFlagOn();
  m_BackProjectionFilter->SetInPlace(true);
  m_BackProjectionFilter->SetReleaseDataFlag(this->GetSupportMask().IsNotNull() || (m_Gamma != 0) || (m_Tikhonov != 0));

  // Update output information on the last filter of the pipeline
  m_FloatingOutputPointer->UpdateOutputInformation();
  this->GetOutput()->CopyInformation(m_FloatingOutputPointer);
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
void
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::GenerateData()
{
  // Execute Pipeline
  m_FloatingOutputPointer->Update();
  this->GraftOutput(m_FloatingOutputPointer);
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
template <typename ImageType>
typename std::enable_if<std::is_same<TSingleComponentImage, ImageType>::value, ImageType>::type::Pointer
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::
  ConnectGradientRegularization()
{
  m_LaplacianFilter = rtk::LaplacianImageFilter<TOutputImage, GradientImageType>::New();
  m_LaplacianFilter->SetInput(m_FloatingInputPointer);
  m_MultiplyLaplacianFilter = MultiplyFilterType::New();
  m_MultiplyLaplacianFilter->SetInput1(m_LaplacianFilter->GetOutput());
  // Set "-1.0*gamma" because we need to perform "-1.0*Laplacian"
  // for correctly applying quadratic regularization || grad f ||_2^2
  m_MultiplyLaplacianFilter->SetConstant2(-1.0 * m_Gamma);

  m_AddLaplacianFilter = AddFilterType::New();
  m_AddLaplacianFilter->SetInput(0, m_BackProjectionFilter->GetOutput());
  if (this->GetLocalRegularizationWeights().IsNotNull())
  {
    m_LaplacianFilter->SetInput("Weights",
                                const_cast<TWeightsImage *>(this->GetLocalRegularizationWeights().GetPointer()));
  }
  m_AddLaplacianFilter->SetInput(1, m_MultiplyLaplacianFilter->GetOutput());

  return m_AddLaplacianFilter->GetOutput();
}

template <typename TOutputImage, typename TSingleComponentImage, typename TWeightsImage>
template <typename ImageType>
typename std::enable_if<!std::is_same<TSingleComponentImage, ImageType>::value, ImageType>::type::Pointer
ReconstructionConjugateGradientOperator<TOutputImage, TSingleComponentImage, TWeightsImage>::
  ConnectGradientRegularization()
{
  itkWarningMacro(<< "Gradient regularization is not enabled for vector images, assuming Gamma 0");
  return m_FloatingOutputPointer;
}

} // namespace rtk


#endif
