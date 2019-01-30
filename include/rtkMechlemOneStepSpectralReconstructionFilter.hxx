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
  m_NumberOfProjectionsPerSubset=0;
  m_NumberOfSubsets=1;
  m_ResetNesterovEvery = itk::NumericTraits<int>::max();
  m_NumberOfProjections=0;
  m_RegularizationWeights.Fill(0);
  m_RegularizationRadius.Fill(0);

  // Create the filters
  m_ExtractPhotonCountsFilter = ExtractPhotonCountsFilterType::New();
  m_AddGradients = AddFilterType::New();
  m_AddHessians = AddMatrixAndDiagonalFilterType::New();
  m_ProjectionsSource = MaterialProjectionsSourceType::New();
  m_SingleComponentProjectionsSource = SingleComponentImageSourceType::New();
  m_SingleComponentVolumeSource = SingleComponentImageSourceType::New();
  m_GradientsSource = GradientsSourceType::New();
  m_HessiansSource = HessiansSourceType::New();
  m_SQSRegul = SQSRegularizationType::New();
  m_WeidingerForward = WeidingerForwardModelType::New();
  m_NewtonFilter = NewtonFilterType::New();
  m_NesterovFilter = NesterovFilterType::New();
  m_MultiplySupportFilter = MultiplyFilterType::New();

  // Set permanent parameters
  m_ProjectionsSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_SingleComponentProjectionsSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType::ValueType>::ZeroValue());
  m_SingleComponentVolumeSource->SetConstant(itk::NumericTraits<typename TOutputImage::PixelType::ValueType>::One);
  m_GradientsSource->SetConstant(itk::NumericTraits<typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::TGradientsImage::PixelType>::ZeroValue());
  m_HessiansSource->SetConstant(itk::NumericTraits<typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::THessiansImage::PixelType>::ZeroValue());
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
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::SetSupportMask(const SingleComponentImageType* support)
{
  this->SetNthInput(3, const_cast<SingleComponentImageType*>(support));
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
typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::SingleComponentImageType::ConstPointer
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::GetSupportMask()
{
  return static_cast< const SingleComponentImageType* >
         ( this->itk::ProcessObject::GetInput(3) );
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
  typedef typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::SingleComponentImageType TSingleComponent;

  // Declare the pointer
  typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::SingleComponentForwardProjectionFilterType::Pointer fw;

  // Instantiate it
  switch(fwtype)
    {
    case(MechlemOneStepSpectralReconstructionFilter::FP_JOSEPH):
      fw = rtk::JosephForwardProjectionImageFilter<TSingleComponent, TSingleComponent>::New();
      break;
    case(MechlemOneStepSpectralReconstructionFilter::FP_CUDARAYCAST):
      fw = CudaSingleComponentForwardProjectionImageFilterType::New();
      if(IsCPUImage())
        itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
    break;

    default:
      itkGenericExceptionMacro(<< "Unhandled --fp value.");
    }
  return fw;
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::HessiansBackProjectionFilterType::Pointer
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::InstantiateHessiansBackProjectionFilter(int bptype)
{
  // Define the type of image to be back projected
  typedef typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::THessiansImage THessians;

  // Declare the pointer
  typename MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>::HessiansBackProjectionFilterType::Pointer bp;

  // Instantiate it
  switch(bptype)
    {
    case(MechlemOneStepSpectralReconstructionFilter::BP_VOXELBASED):
      bp = rtk::BackProjectionImageFilter<THessians, THessians>::New();
      break;
    case(MechlemOneStepSpectralReconstructionFilter::BP_JOSEPH):
      bp = rtk::JosephBackProjectionImageFilter<THessians, THessians>::New();
      break;
    case(MechlemOneStepSpectralReconstructionFilter::BP_CUDAVOXELBASED):
      bp = CudaHessiansBackProjectionImageFilterType::New();
      if(IsCPUImage())
        itkGenericExceptionMacro(<< "The program has not been compiled with cuda option");
      break;
    case(MechlemOneStepSpectralReconstructionFilter::BP_CUDARAYCAST):
      itkGenericExceptionMacro(<< "The CUDA ray cast back projector can currently not handle vector images");
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

  // Input 3 is the support (optional)
  typename SingleComponentImageType::Pointer inputPtr3 =
          const_cast< SingleComponentImageType * >( this->GetSupportMask().GetPointer() );
  if ( !inputPtr3 )
      return;
  inputPtr3->SetRequestedRegion( inputPtr0->GetRequestedRegion() );
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::GenerateOutputInformation()
{
  typename TPhotonCounts::RegionType largest = this->GetInputPhotonCounts()->GetLargestPossibleRegion();
  m_NumberOfProjections = largest.GetSize()[TPhotonCounts::ImageDimension - 1];

  // Pre-compute the number of projections in each subset
  m_NumberOfProjectionsInSubset.clear();
  m_NumberOfProjectionsPerSubset = itk::Math::ceil( (float) m_NumberOfProjections / (float) m_NumberOfSubsets);
  for (int s=0; s<m_NumberOfSubsets; s++)
    m_NumberOfProjectionsInSubset.push_back(std::min(m_NumberOfProjectionsPerSubset, m_NumberOfProjections - s * m_NumberOfProjectionsPerSubset));

  // Compute the extract filter's initial extract region
  typename TPhotonCounts::RegionType extractionRegion = largest;
  extractionRegion.SetSize(TPhotonCounts::ImageDimension - 1, m_NumberOfProjectionsInSubset[0]);
  extractionRegion.SetIndex(TPhotonCounts::ImageDimension - 1, 0);

  // Set runtime connections. Links with the forward and back projection filters should be set here,
  // since those filters are not instantiated by the constructor, but by
  // a call to SetForwardProjectionFilter() and SetBackProjectionFilter()
  m_ExtractPhotonCountsFilter->SetInput(this->GetInputPhotonCounts());

  m_ForwardProjectionFilter->SetInput(0, m_ProjectionsSource->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, this->GetInputMaterialVolumes());

  m_SingleComponentForwardProjectionFilter->SetInput(0, m_SingleComponentProjectionsSource->GetOutput());
  m_SingleComponentForwardProjectionFilter->SetInput(1, m_SingleComponentVolumeSource->GetOutput());

  m_WeidingerForward->SetInputMaterialProjections(m_ForwardProjectionFilter->GetOutput());
  m_WeidingerForward->SetInputPhotonCounts(m_ExtractPhotonCountsFilter->GetOutput());
  m_WeidingerForward->SetInputSpectrum(this->GetInputSpectrum());
  m_WeidingerForward->SetInputProjectionsOfOnes(m_SingleComponentForwardProjectionFilter->GetOutput());

  m_GradientsBackProjectionFilter->SetInput(0, m_GradientsSource->GetOutput());
  m_GradientsBackProjectionFilter->SetInput(1, m_WeidingerForward->GetOutput1());

  m_HessiansBackProjectionFilter->SetInput(0, m_HessiansSource->GetOutput());
  m_HessiansBackProjectionFilter->SetInput(1, m_WeidingerForward->GetOutput2());

  m_SQSRegul->SetInput(this->GetInputMaterialVolumes());

  m_AddGradients->SetInput1(m_SQSRegul->GetOutput(0));
  m_AddGradients->SetInput2(m_GradientsBackProjectionFilter->GetOutput());

  m_AddHessians->SetInputDiagonal(m_SQSRegul->GetOutput(1));
  m_AddHessians->SetInputMatrix(m_HessiansBackProjectionFilter->GetOutput());

  m_NewtonFilter->SetInputGradient(m_AddGradients->GetOutput());
  m_NewtonFilter->SetInputHessian(m_AddHessians->GetOutput());

  m_NesterovFilter->SetInput(0, this->GetInputMaterialVolumes());
  m_NesterovFilter->SetInput(1, m_NewtonFilter->GetOutput());

  typename TOutputImage::Pointer lastOutput = m_NesterovFilter->GetOutput();
  if(this->GetSupportMask().GetPointer() != ITK_NULLPTR)
    {
    m_MultiplySupportFilter->SetInput1( m_NesterovFilter->GetOutput() );
    m_MultiplySupportFilter->SetInput2( this->GetSupportMask() );
    lastOutput = m_MultiplySupportFilter->GetOutput();
    }

  // Set information for the extract filter and the sources
  m_ExtractPhotonCountsFilter->SetExtractionRegion(extractionRegion);
  m_ExtractPhotonCountsFilter->UpdateOutputInformation();
  m_SingleComponentProjectionsSource->SetInformationFromImage(m_ExtractPhotonCountsFilter->GetOutput());
  m_ProjectionsSource->SetInformationFromImage(m_ExtractPhotonCountsFilter->GetOutput());
  m_SingleComponentVolumeSource->SetInformationFromImage(this->GetInputMaterialVolumes());
  m_GradientsSource->SetInformationFromImage(this->GetInputMaterialVolumes());
  m_HessiansSource->SetInformationFromImage(this->GetInputMaterialVolumes());

  // For the same reason, set geometry now
  m_ForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_SingleComponentForwardProjectionFilter->SetGeometry(this->m_Geometry);
  m_GradientsBackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());
  m_HessiansBackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());

  // Set regularization parameters
  m_SQSRegul->SetRegularizationWeights(m_RegularizationWeights);
  m_SQSRegul->SetRadius(m_RegularizationRadius);

  // Have the last filter calculate its output information
  lastOutput->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( lastOutput );
}

template< class TOutputImage, class TPhotonCounts, class TSpectrum>
void
MechlemOneStepSpectralReconstructionFilter< TOutputImage, TPhotonCounts, TSpectrum>
::GenerateData()
{
  // Run the iteration loop
  typename TOutputImage::Pointer Next_Zk;
  for(int iter = 0; iter < m_NumberOfIterations; iter++)
    {
    for (int subset = 0; subset < m_NumberOfSubsets; subset++)
      {
      // Initialize Nesterov filter
      int k = iter*m_NumberOfSubsets + subset;
      if(k%m_ResetNesterovEvery == 0)
        {
        int r = m_NumberOfIterations*m_NumberOfSubsets-k;
        m_NesterovFilter->SetNumberOfIterations(std::min(m_ResetNesterovEvery, r));
        }

      // Starting from the second subset, or the second iteration
      // if there is only one subset, plug the output
      // of Nesterov back as input of the forward projection
      // The Nesterov filter itself doesn't need its output
      // plugged back as input, since it stores intermediate
      // images that contain all the required data. It only
      // needs the new update from rtkGetNewtonUpdateImageFilter
      if ((iter + subset) >0)
        {
        Next_Zk->DisconnectPipeline();
        m_ForwardProjectionFilter->SetInput(1, Next_Zk);
        m_SQSRegul->SetInput(Next_Zk);
        m_NesterovFilter->SetInput(Next_Zk);
        }

      // Set the extract filter's region
      typename TPhotonCounts::RegionType extractionRegion = this->GetInputPhotonCounts()->GetLargestPossibleRegion();
      extractionRegion.SetSize(TPhotonCounts::ImageDimension - 1, m_NumberOfProjectionsInSubset[subset]);
      extractionRegion.SetIndex(TPhotonCounts::ImageDimension - 1, subset * m_NumberOfProjectionsPerSubset);
      m_ExtractPhotonCountsFilter->SetExtractionRegion(extractionRegion);
      m_ExtractPhotonCountsFilter->UpdateOutputInformation();

      // Set the projection sources accordingly
      m_SingleComponentProjectionsSource->SetInformationFromImage(m_ExtractPhotonCountsFilter->GetOutput());
      m_ProjectionsSource->SetInformationFromImage(m_ExtractPhotonCountsFilter->GetOutput());

      // Update the most downstream filter
      if(this->GetSupportMask().GetPointer() != ITK_NULLPTR)
        {
        m_MultiplySupportFilter->Update();
        Next_Zk = m_MultiplySupportFilter->GetOutput();
        }
      else
        {
        m_NesterovFilter->Update();
        Next_Zk = m_NesterovFilter->GetOutput();
        }
      }

    }
  this->GraftOutput( Next_Zk );
}

}// end namespace


#endif
