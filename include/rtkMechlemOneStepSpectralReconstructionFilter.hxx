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

#ifndef rtkMechlemOneStepSpectralReconstructionFilter_hxx
#define rtkMechlemOneStepSpectralReconstructionFilter_hxx

#include "rtkMechlemOneStepSpectralReconstructionFilter.h"

namespace rtk
{

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::MechlemOneStepSpectralReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(3);

  // Set the default values of member parameters
  m_NumberOfIterations=3;
//  m_IterationCosts=false;
//  m_Gamma = 0;
//  m_Regularized = false;

  // Create the filters
  m_ProjectionsSource = MaterialProjectionsSourceType::New();
  m_SingleComponentProjectionsSource = SingleComponentImageSourceType::New();
  m_SingleComponentVolumeSource = SingleComponentImageSourceType::New();
  m_GradientsSource = GradientsSourceType::New();
  m_HessiansSource = HessiansSourceType::New();
  m_WeidingerForward = WeidingerForwardModelType::New();
  m_NewtonFilter = NewtonFilterType::New();
  m_NesterovFilter = NesterovFilterType::New();

  // Set permanent parameters
  m_ProjectionsSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType>::Zero);
  m_SingleComponentProjectionsSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType::ValueType>::Zero);
  m_SingleComponentVolumeSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType::ValueType>::One);
  m_GradientsSource->SetConstant(itk::NumericTraits<typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::TGradientsImage::PixelType>::Zero);
  m_HessiansSource->SetConstant(itk::NumericTraits<typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::THessiansImage::PixelType>::Zero);
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::SetInputMaterialVolumes(const TOutputImage* materialVolumes)
{
  this->SetNthInput(0, const_cast<TOutputImage*>(materialVolumes));
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::SetInputPhotonCounts(const TPhotonCounts* photonCounts)
{
  this->SetNthInput(1, const_cast<TPhotonCounts*>(photonCounts));
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::SetInputSpectrum(const TSpectrum* spectrum)
{
  this->SetNthInput(2, const_cast<TSpectrum*>(spectrum));
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
typename TOutputImage::ConstPointer
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::GetInputMaterialVolumes()
{
  return static_cast< const TOutputImage * >
         ( this->itk::ProcessObject::GetInput(0) );
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
typename TPhotonCounts::ConstPointer
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::GetInputPhotonCounts()
{
  return static_cast< const TPhotonCounts * >
         ( this->itk::ProcessObject::GetInput(1) );
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
typename TSpectrum::ConstPointer
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::GetInputSpectrum()
{
  return static_cast< const TSpectrum * >
         ( this->itk::ProcessObject::GetInput(2) );
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::SetForwardProjectionFilter (ForwardProjectionType _arg)
{
  if( _arg != this->GetForwardProjectionFilter() )
    {
    Superclass::SetForwardProjectionFilter( _arg );
    m_ForwardProjectionFilter = this->InstantiateForwardProjectionFilter( _arg ); // The multi-component one
    m_SingleComponentForwardProjectionFilter = InstantiateSingleComponentForwardProjectionFilter( _arg ); // The single-component one
    }
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::SetBackProjectionFilter (BackProjectionType _arg)
{
  if( _arg != this->GetBackProjectionFilter() )
    {
    Superclass::SetBackProjectionFilter( _arg );
    m_GradientsBackProjectionFilter = this->InstantiateBackProjectionFilter( _arg );
    m_HessiansBackProjectionFilter = this->InstantiateHessiansBackProjectionFilter( _arg );
    }
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::SingleComponentForwardProjectionFilterType::Pointer
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::InstantiateSingleComponentForwardProjectionFilter (int fwtype)
{
  // Define the type of image to be back projected
  typedef typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::TSingleComponentImage TSingleComponent;

  // Declare the pointer
  typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::SingleComponentForwardProjectionFilterType::Pointer fw;
  switch(fwtype)
    {
    case(0):
      fw = rtk::JosephForwardProjectionImageFilter<TSingleComponent, TSingleComponent>::New();
    break;
    case(1):
    #ifdef RTK_USE_CUDA
      fw = rtk::CudaForwardProjectionImageFilter<TSingleComponent, TSingleComponent>::New();
    #else
      itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
    #endif
    break;

    default:
      itkGenericExceptionMacro(<< "Unhandled --fp value.");
    }
  return fw;
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::HessiansBackProjectionFilterType::Pointer
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::InstantiateHessiansBackProjectionFilter (int bptype)
{
  // Define the type of image to be back projected
  typedef typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::THessiansImage THessians;

  // Declare the pointer
  typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::HessiansBackProjectionFilterType::Pointer bp;
  switch(bptype)
    {
    case(0):
      bp = rtk::BackProjectionImageFilter<THessians,THessians>::New();
      break;
    case(1):
      bp = rtk::JosephBackProjectionImageFilter<THessians,THessians>::New();
      break;
    case(2):
    #ifdef RTK_USE_CUDA
      // The current CUDA implementation only allows one component
//      bp = rtk::CudaBackProjectionImageFilter::New();
    #else
      itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
    #endif
    break;
//    case(3):
//      bp = rtk::NormalizedJosephBackProjectionImageFilter<THessians,THessians>::New();
//      break;
    case(4):
    #ifdef RTK_USE_CUDA
      // The current CUDA implementation only allows one component
//      bp = rtk::CudaRayCastBackProjectionImageFilter::New();
    #else
      itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
    #endif
      break;
    default:
      itkGenericExceptionMacro(<< "Unhandled --bp value.");
    }
  return bp;
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::SetMaterialAttenuations(const MaterialAttenuationsType & matAtt)
{
  m_WeidingerForward->SetMaterialAttenuations(matAtt);
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::SetBinnedDetectorResponse(const BinnedDetectorResponseType & detResp)
{
  m_WeidingerForward->SetBinnedDetectorResponse(detResp);
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::GenerateInputRequestedRegion()
{
  // Input 0 is the material volumes we update
  typename TOutputImage::Pointer inputPtr0 =
          const_cast< TOutputImage * >( this->GetInputMaterialVolumes().GetPointer() );
  if ( !inputPtr0 )
      return;
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the photon counts
  typename TPhotonCounts::Pointer inputPtr1 =
          const_cast< TPhotonCounts * >( this->GetInputPhotonCounts().GetPointer() );
  if ( !inputPtr1 )
      return;
  inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );

  // Input 2 is the incident spectrum
  typename TSpectrum::Pointer inputPtr2 =
          const_cast< TSpectrum * >( this->GetInputSpectrum().GetPointer() );
  if ( !inputPtr2 )
      return;
  inputPtr2->SetRequestedRegion( inputPtr2->GetLargestPossibleRegion() );
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::GenerateOutputInformation()
{
  // Set runtime connections. Links with the forward and back projection filters should be set here,
  // since those filters are not instantiated by the constructor, but by
  // a call to SetForwardProjectionFilter() and SetBackProjectionFilter()
  m_ForwardProjectionFilter->SetInput(0, m_ProjectionsSource->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, this->GetInputMaterialVolumes());

  m_SingleComponentForwardProjectionFilter->SetInput(0, m_SingleComponentProjectionsSource->GetOutput());
  m_SingleComponentForwardProjectionFilter->SetInput(1, m_SingleComponentVolumeSource->GetOutput());

  m_WeidingerForward->SetInputMaterialProjections(m_ForwardProjectionFilter->GetOutput());
  m_WeidingerForward->SetInputPhotonCounts(this->GetInputPhotonCounts());
  m_WeidingerForward->SetInputSpectrum(this->GetInputSpectrum());
  m_WeidingerForward->SetInputProjectionsOfOnes(m_SingleComponentForwardProjectionFilter->GetOutput());

  m_GradientsBackProjectionFilter->SetInput(0, m_GradientsSource->GetOutput());
  m_GradientsBackProjectionFilter->SetInput(1, m_WeidingerForward->GetOutput1());

  m_HessiansBackProjectionFilter->SetInput(0, m_HessiansSource->GetOutput());
  m_HessiansBackProjectionFilter->SetInput(1, m_WeidingerForward->GetOutput2());

  m_NewtonFilter->SetInputGradient(m_GradientsBackProjectionFilter->GetOutput());
  m_NewtonFilter->SetInputHessian(m_HessiansBackProjectionFilter->GetOutput());

  m_NesterovFilter->SetInput(0, this->GetInputMaterialVolumes());
  m_NesterovFilter->SetInput(1, m_NewtonFilter->GetOutput());

  // Set information for the sources
  m_SingleComponentProjectionsSource->SetInformationFromImage(this->GetInputPhotonCounts());
  m_ProjectionsSource->SetInformationFromImage(this->GetInputPhotonCounts());
  m_SingleComponentVolumeSource->SetInformationFromImage(this->GetInputMaterialVolumes());
  m_GradientsSource->SetInformationFromImage(this->GetInputMaterialVolumes());
  m_HessiansSource->SetInformationFromImage(this->GetInputMaterialVolumes());

  // For the same reason, set geometry now
  m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_SingleComponentForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_GradientsBackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());
  m_HessiansBackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());

  // Set memory management parameters


  // Have the last filter calculate its output information
  m_NesterovFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_NesterovFilter->GetOutput() );
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::GenerateData()
{
  // Nesterov filter needs its input to be updated before it is initialized
  // since it allocates its internal images to the same size as the input
  m_NesterovFilter->UpdateOutputInformation();

  // Initialize Nesterov filter
  m_NesterovFilter->SetNumberOfIterations(m_NumberOfIterations+1);
  m_NesterovFilter->ResetIterations();

  typename TOutputImage::Pointer NextAlpha_k;

  for(unsigned int iter = 0; iter < m_NumberOfIterations; iter++)
    {
    // Starting from the second iteration, plug the output
    // of Nesterov back as input of the forward projection
    // The Nesterov filter itself doesn't need its output
    // plugged back as input, since it stores intermediate
    // images that contain all the required data. It only
    // needs the new update from rtkGetNewtonUpdateImageFilter
    if (iter>0)
      {
      NextAlpha_k = m_NesterovFilter->GetOutput();
      NextAlpha_k->DisconnectPipeline();
      m_ForwardProjectionFilter->SetInput(1, NextAlpha_k);
      }

    // Update the most downstream filter
    m_NesterovFilter->Update();
    }
  this->GraftOutput( m_NesterovFilter->GetOutput() );
}

}// end namespace


#endif
