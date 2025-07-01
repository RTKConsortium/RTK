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

#ifndef rtkFourDConjugateGradientConeBeamReconstructionFilter_hxx
#define rtkFourDConjugateGradientConeBeamReconstructionFilter_hxx

#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"

#include <algorithm>

#include <itkImageFileWriter.h>
#include <itkIterationReporter.h>

namespace rtk
{

template <class VolumeSeriesType, class ProjectionStackType>
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::
  FourDConjugateGradientConeBeamReconstructionFilter()
  : m_IterationReporter(this, 0, 1)
{
  this->SetNumberOfRequiredInputs(2); // 4D sequence, projections

  // Set the default values of member parameters
  m_NumberOfIterations = 3;
  m_CudaConjugateGradient = false; // 4D volumes of usual size only fit on the largest GPUs

  // Create the filters
  m_CGOperator = CGOperatorFilterType::New();
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
  m_ProjStackToFourDFilter = ProjStackToFourDFilterType::New();
  m_DisplacedDetectorFilter = DisplacedDetectorFilterType::New();

  // Set parameters
  m_DisplacedDetectorFilter->SetPadOnTruncatedSide(false);
  m_DisableDisplacedDetectorFilter = false;

  // Memory management options
  m_DisplacedDetectorFilter->ReleaseDataFlagOn();
  m_ProjStackToFourDFilter->ReleaseDataFlagOn();
}

template <class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::SetInputVolumeSeries(
  const VolumeSeriesType * VolumeSeries)
{
  this->SetNthInput(0, const_cast<VolumeSeriesType *>(VolumeSeries));
}

template <class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::SetInputProjectionStack(
  const ProjectionStackType * Projections)
{
  this->SetNthInput(1, const_cast<ProjectionStackType *>(Projections));
}

template <class VolumeSeriesType, class ProjectionStackType>
typename VolumeSeriesType::ConstPointer
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::GetInputVolumeSeries()
{
  return static_cast<const VolumeSeriesType *>(this->itk::ProcessObject::GetInput(0));
}

template <class VolumeSeriesType, class ProjectionStackType>
typename ProjectionStackType::ConstPointer
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::GetInputProjectionStack()
{
  return static_cast<const ProjectionStackType *>(this->itk::ProcessObject::GetInput(1));
}

template <class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::SetWeights(
  const itk::Array2D<float> _arg)
{
  m_ProjStackToFourDFilter->SetWeights(_arg);
  m_CGOperator->SetWeights(_arg);
  this->Modified();
}

template <class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::SetSignal(
  const std::vector<double> signal)
{
  m_ProjStackToFourDFilter->SetSignal(signal);
  m_CGOperator->SetSignal(signal);
  this->m_Signal = signal;
  this->Modified();
}

template <class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::GenerateOutputInformation()
{
  // Set the Conjugate Gradient filter (either on CPU or GPU depending on user's choice)
  if (m_CudaConjugateGradient)
  {
    if (std::is_same<VolumeSeriesType, CPUVolumeSeriesType>::value)
      itkGenericExceptionMacro(<< "CudaConjugateGradient option only available with itk::CudaImage.");
    m_ConjugateGradientFilter = CudaConjugateGradientImageFilterType::New();
  }

  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());

  // Set runtime connections
  m_CGOperator->SetInputProjectionStack(this->GetInputProjectionStack());
  m_ConjugateGradientFilter->SetX(this->GetInputVolumeSeries());
  m_DisplacedDetectorFilter->SetInput(this->GetInputProjectionStack());

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_ProjStackToFourDFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
  m_ProjStackToFourDFilter->SetInputProjectionStack(m_DisplacedDetectorFilter->GetOutput());
  m_ConjugateGradientFilter->SetB(m_ProjStackToFourDFilter->GetOutput());

  // For the same reason, set geometry now
  m_CGOperator->SetGeometry(this->GetGeometry());
  m_ProjStackToFourDFilter->SetGeometry(this->GetGeometry());
  m_DisplacedDetectorFilter->SetGeometry(this->GetGeometry());

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_NumberOfIterations);
  m_DisplacedDetectorFilter->SetDisable(m_DisableDisplacedDetectorFilter);
  m_CGOperator->SetDisableDisplacedDetectorFilter(m_DisableDisplacedDetectorFilter);

  // Set forward projection filter
  m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter(this->m_CurrentForwardProjectionConfiguration);
  // Pass the ForwardProjection filter to the conjugate gradient operator
  m_CGOperator->SetForwardProjectionFilter(m_ForwardProjectionFilter);
  if (this->m_CurrentForwardProjectionConfiguration ==
      ForwardProjectionType::FP_CUDARAYCAST) // The forward projection filter runs on GPU. It is most efficient to also
                                             // run the interpolation on GPU, and to use GPU constant image sources
  {
    m_CGOperator->SetUseCudaInterpolation(true);
    m_CGOperator->SetUseCudaSources(true);
  }

  // Set back projection filter
  m_BackProjectionFilter = this->InstantiateBackProjectionFilter(this->m_CurrentBackProjectionConfiguration);
  m_CGOperator->SetBackProjectionFilter(m_BackProjectionFilter);

  m_BackProjectionFilterForB = this->InstantiateBackProjectionFilter(this->m_CurrentBackProjectionConfiguration);
  // Pass the backprojection filter to the conjugate gradient operator and to the filter generating the B of AX=B
  m_ProjStackToFourDFilter->SetBackProjectionFilter(m_BackProjectionFilterForB);
  if (this->m_CurrentBackProjectionConfiguration ==
      BackProjectionType::BP_CUDAVOXELBASED) // The back projection filter runs on GPU. It is most efficient to also run
                                             // the splat on GPU, and to use GPU constant image sources
  {
    m_CGOperator->SetUseCudaSplat(true);
    m_CGOperator->SetUseCudaSources(true);
    m_ProjStackToFourDFilter->SetUseCudaSplat(true);
    m_ProjStackToFourDFilter->SetUseCudaSources(true);
  }

  // Have the last filter calculate its output information
  m_ConjugateGradientFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation(m_ConjugateGradientFilter->GetOutput());
}


template <class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType,
                                                   ProjectionStackType>::GenerateInputRequestedRegion()
{
  // Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  this->m_ProjStackToFourDFilter->PropagateRequestedRegion(this->m_ProjStackToFourDFilter->GetOutput());
}

template <class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::GenerateData()
{
  auto callbackCommand = itk::MemberCommand<Self>::New();
  callbackCommand->SetCallbackFunction(this, &Self::ReportProgress);
  m_ConjugateGradientFilter->AddObserver(itk::IterationEvent(), callbackCommand);

  m_ProjStackToFourDFilter->Update();

  if (!m_CudaConjugateGradient)
    this->m_ProjStackToFourDFilter->GetOutput()->GetBufferPointer();

  m_ConjugateGradientFilter->Update();

  // Simply grafting the output of m_ConjugateGradientFilter to the main output
  // is sufficient in most cases, but when this output is then disconnected and replugged,
  // several images end up having the same CudaDataManager. The following solution is a
  // workaround for this problem
  typename VolumeSeriesType::Pointer pimg = m_ConjugateGradientFilter->GetOutput();
  pimg->DisconnectPipeline();

  this->GraftOutput(pimg);
}

template <class VolumeSeriesType, class ProjectionStackType>
void
FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>::ReportProgress(
  itk::Object *            caller,
  const itk::EventObject & event)
{
  {
    if (!itk::IterationEvent().CheckEvent(&event))
    {
      return;
    }
    auto * cgCaller = dynamic_cast<rtk::ConjugateGradientImageFilter<VolumeSeriesType> *>(caller);
    if (cgCaller)
    {
      this->GraftOutput(cgCaller->GetOutput());
      m_IterationReporter.CompletedStep();
    }
  }
}

} // end namespace rtk

#endif // rtkFourDConjugateGradientConeBeamReconstructionFilter_hxx
