#ifndef __rtkADMMTotalVariationConeBeamReconstructionFilter_txx
#define __rtkADMMTotalVariationConeBeamReconstructionFilter_txx

#include "rtkADMMTotalVariationConeBeamReconstructionFilter.h"

#include "rtkJosephForwardProjectionImageFilter.h"

namespace rtk
{

template< typename TOutputImage>
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage>::ADMMTotalVariationConeBeamReconstructionFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set the default values of member parameters
  m_Alpha=1;
  m_Beta=1;
  m_AL_iterations=10;
  m_CG_iterations=3;
  m_MeasureExecutionTimes=false;
  m_CurrentBackProjectionConfiguration = -1;
  m_CurrentForwardProjectionConfiguration = -1;

  // Create the filters
  m_ZeroMultiplyVolumeFilter = MultiplyVolumeFilterType::New();
  m_ZeroMultiplyGradientFilter = MultiplyGradientFilterType::New();
  m_SubtractFilter1 = SubtractGradientsFilterType::New();
  m_SubtractFilter2 = SubtractGradientsFilterType::New();
  m_MultiplyFilter = MultiplyVolumeFilterType::New();
  m_GradientFilter1 = ImageGradientFilterType::New();
  m_GradientFilter2 = ImageGradientFilterType::New();
  m_AddVolumeFilter = AddVolumeFilterType::New();
  m_AddGradientsFilter = AddGradientsFilterType::New();
  m_DivergenceFilter = ImageDivergenceFilterType::New();
  m_ConjugateGradientFilter = ConjugateGradientFilterType::New();
  m_SoftThresholdFilter = SoftThresholdTVFilterType::New();
  m_CGOperator = CGOperatorFilterType::New();
  m_ConjugateGradientFilter->SetA(m_CGOperator.GetPointer());

  // Set permanent connections
  m_ZeroMultiplyGradientFilter->SetInput1(m_GradientFilter1->GetOutput());
  m_AddGradientsFilter->SetInput1(m_ZeroMultiplyGradientFilter->GetOutput());
  m_AddGradientsFilter->SetInput2(m_GradientFilter1->GetOutput());
  m_DivergenceFilter->SetInput(m_AddGradientsFilter->GetOutput());
  m_MultiplyFilter->SetInput1( m_DivergenceFilter->GetOutput() );
  m_AddVolumeFilter->SetInput2(m_MultiplyFilter->GetOutput());
  m_ConjugateGradientFilter->SetB(m_AddVolumeFilter->GetOutput());
  m_ConjugateGradientFilter->SetNumberOfIterations(m_CG_iterations);
  m_ConjugateGradientFilter->SetMeasureExecutionTimes(m_MeasureExecutionTimes);
  m_GradientFilter2->SetInput(m_ConjugateGradientFilter->GetOutput());
  m_SubtractFilter1->SetInput1(m_GradientFilter2->GetOutput());
  m_SubtractFilter1->SetInput2(m_ZeroMultiplyGradientFilter->GetOutput());
  m_SoftThresholdFilter->SetInput(m_SubtractFilter1->GetOutput());
  m_SubtractFilter2->SetInput1(m_SoftThresholdFilter->GetOutput());
  m_SubtractFilter2->SetInput2(m_SubtractFilter1->GetOutput());

  // Set permanent parameters
  m_ZeroMultiplyVolumeFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());
  m_ZeroMultiplyGradientFilter->SetConstant2(itk::NumericTraits<typename TOutputImage::PixelType>::ZeroValue());

  // Set memory management parameters
  m_ZeroMultiplyVolumeFilter->ReleaseDataFlagOn();
  m_ZeroMultiplyGradientFilter->ReleaseDataFlagOn();
  m_GradientFilter1->ReleaseDataFlagOn();
  m_AddGradientsFilter->ReleaseDataFlagOn();
  m_DivergenceFilter->ReleaseDataFlagOn();
  m_MultiplyFilter->ReleaseDataFlagOn();
  m_AddVolumeFilter->ReleaseDataFlagOn();
  m_ConjugateGradientFilter->ReleaseDataFlagOff(); // Output is f_k+1
  m_GradientFilter2->ReleaseDataFlagOn();
  m_SubtractFilter1->ReleaseDataFlagOff(); // Output used in two filters
  m_SoftThresholdFilter->ReleaseDataFlagOff(); // Output is g_k+1
  m_SubtractFilter2->ReleaseDataFlagOff(); //Output is d_k+1
}

template< typename TOutputImage>
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage>
::ConfigureForwardProjection (int _arg)
{
  switch(_arg)
    {
    case(0):
      m_ForwardProjectionFilter = rtk::JosephForwardProjectionImageFilter<TOutputImage, TOutputImage>::New();
    break;
    case(1):
    #if CUDA_FOUND
      m_ForwardProjectionFilter = rtk::CudaForwardProjectionImageFilter::New();
    #else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
    #endif
    break;
    case(2):
      m_ForwardProjectionFilter = rtk::RayCastInterpolatorForwardProjectionImageFilter<TOutputImage, TOutputImage>::New();
    break;

    default:
      std::cerr << "Unhandled --method value." << std::endl;
    }
  m_CGOperator->SetForwardProjectionFilter( m_ForwardProjectionFilter );

  if (m_CurrentForwardProjectionConfiguration != _arg)
    {
    this->Modified();
    m_CGOperator->Modified();
    }
}

template< typename TOutputImage>
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage>
::ConfigureBackProjection (int _arg)
{
  switch(_arg)
    {
    case(0):
      m_BackProjectionFilterForConjugateGradient  = rtk::BackProjectionImageFilter<TOutputImage, TOutputImage>::New();
      m_BackProjectionFilter = rtk::BackProjectionImageFilter<TOutputImage, TOutputImage>::New();
      break;
    case(1):
      m_BackProjectionFilterForConjugateGradient = rtk::JosephBackProjectionImageFilter<TOutputImage, TOutputImage>::New();
      m_BackProjectionFilter = rtk::JosephBackProjectionImageFilter<TOutputImage, TOutputImage>::New();
      break;
    case(2):
    #if CUDA_FOUND
      m_BackProjectionFilterForConjugateGradient = rtk::CudaBackProjectionImageFilter::New();
      m_BackProjectionFilter = rtk::CudaBackProjectionImageFilter::New();
    #else
      std::cerr << "The program has not been compiled with cuda option" << std::endl;
    #endif
    break;

    default:
      std::cerr << "Unhandled --bp value." << std::endl;
    }
  m_CGOperator->SetBackProjectionFilter( m_BackProjectionFilterForConjugateGradient );

  if (m_CurrentBackProjectionConfiguration != _arg)
    {
    this->Modified();
    m_CGOperator->Modified();
    }
}

template< typename TOutputImage>
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage>
::SetBetaForCurrentIteration(int iter){

    float currentBeta = m_Beta * (iter+1) / m_AL_iterations;

    m_CGOperator->SetBeta(currentBeta);
    m_SoftThresholdFilter->SetThreshold(m_Alpha/(2 * currentBeta));
    m_MultiplyFilter->SetConstant2( (const float) currentBeta);
}

template< typename TOutputImage>
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage>
::GenerateInputRequestedRegion()
{
    // Input 0 is the volume we update
    typename Superclass::InputImagePointer inputPtr0 =
            const_cast< TOutputImage * >( this->GetInput(0) );
    if ( !inputPtr0 )
        return;
    inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

    // Input 1 is the stack of projections to backproject
    typename Superclass::InputImagePointer  inputPtr1 =
            const_cast< TOutputImage * >( this->GetInput(1) );
    if ( !inputPtr1 )
        return;
    inputPtr1->SetRequestedRegion( inputPtr1->GetLargestPossibleRegion() );

}

template< typename TOutputImage>
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_GradientFilter1->SetInput(this->GetInput(0));
  m_ZeroMultiplyVolumeFilter->SetInput1(this->GetInput(0));
  m_CGOperator->SetInput(1, this->GetInput(1));
  m_ConjugateGradientFilter->SetX(this->GetInput(0));
  m_MultiplyFilter->SetConstant2( m_Beta );

  // Links with the m_BackProjectionFilter should be set here and not
  // in the constructor, as m_BackProjectionFilter is set at runtime
  m_BackProjectionFilter->SetInput(0, m_ZeroMultiplyVolumeFilter->GetOutput());
  m_BackProjectionFilter->SetInput(1, this->GetInput(1));
  m_AddVolumeFilter->SetInput1(m_BackProjectionFilter->GetOutput());

  // For the same reason, set geometry now
  m_CGOperator->SetGeometry(this->m_Geometry);
  m_BackProjectionFilter->SetGeometry(this->m_Geometry.GetPointer());

  // Set runtime parameters
  m_ConjugateGradientFilter->SetNumberOfIterations(this->m_CG_iterations);

  // Have the last filter calculate its output information
  m_SubtractFilter2->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation( m_SubtractFilter2->GetOutput() );
}

template< typename TOutputImage>
void
ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage>
::GenerateData()
{
  itk::TimeProbe ADMMTimeProbe;
  if(m_MeasureExecutionTimes)
    {
    std::cout << "Starting ADMM initialization" << std::endl;
    ADMMTimeProbe.Start();
    }

  float PreviousTimeTotal, TimeDifference;
  PreviousTimeTotal = 0;
  TimeDifference = 0;
  if(m_MeasureExecutionTimes)
    {
    ADMMTimeProbe.Stop();
    std::cout << "ADMM initialization took " << ADMMTimeProbe.GetTotal() << ' ' << ADMMTimeProbe.GetUnit() << std::endl;
    PreviousTimeTotal = ADMMTimeProbe.GetTotal();
    }

  for(unsigned int iter=0; iter < m_AL_iterations; iter++)
    {
    SetBetaForCurrentIteration(iter);

    if(m_MeasureExecutionTimes)
      {
      std::cout << "Starting ADMM iteration "<< iter << std::endl;
      ADMMTimeProbe.Start();
      }

    // After the first update, we need to use some outputs as inputs
    if(iter>0)
      {
      typename TOutputImage::Pointer f_k_plus_one = m_ConjugateGradientFilter->GetOutput();
      f_k_plus_one->DisconnectPipeline();
      m_ConjugateGradientFilter->SetX(f_k_plus_one);

      typename ImageGradientFilterType::OutputImageType::Pointer g_k_plus_one = m_SoftThresholdFilter->GetOutput();
      g_k_plus_one->DisconnectPipeline();
      m_AddGradientsFilter->SetInput2(g_k_plus_one);

      typename ImageGradientFilterType::OutputImageType::Pointer d_k_plus_one = m_SubtractFilter2->GetOutput();
      d_k_plus_one->DisconnectPipeline();
      m_AddGradientsFilter->SetInput1(d_k_plus_one);
      m_SubtractFilter1->SetInput2(d_k_plus_one);

      // Recreate the links destroyed by DisconnectPipeline
      m_GradientFilter2->SetInput(m_ConjugateGradientFilter->GetOutput());
      m_SubtractFilter2->SetInput1(m_SoftThresholdFilter->GetOutput());
      }

    std::cout << "ADMM iteration " << iter << std::endl;
    m_SubtractFilter2->Update();

    if(m_MeasureExecutionTimes)
      {
      ADMMTimeProbe.Stop();
      TimeDifference = ADMMTimeProbe.GetTotal() - PreviousTimeTotal;
      std::cout << "ADMM iteration " << iter << " took " << TimeDifference << ' ' << ADMMTimeProbe.GetUnit() << std::endl;
      PreviousTimeTotal = ADMMTimeProbe.GetTotal();
      }

    }
  this->GraftOutput( m_ConjugateGradientFilter->GetOutput() );
}

}// end namespace


#endif
