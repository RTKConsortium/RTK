#ifndef __rtkFourDToProjectionStackImageFilter_txx
#define __rtkFourDToProjectionStackImageFilter_txx

#include "rtkFourDToProjectionStackImageFilter.h"

namespace rtk
{

template< typename ProjectionStackType, typename VolumeSeriesType>
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>::FourDToProjectionStackImageFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Set default values
  m_ProjectionNumber = 0;

  // Create the filters that can be created (all but the forward projection filter)
  m_ExtractFilter = ExtractFilterType::New();
  m_PasteFilter = PasteFilterType::New();
  m_InterpolationFilter = InterpolatorFilterType::New();
  m_ConstantSource = ConstantSourceType::New();
  m_ZeroMultiplyFilter = MultiplyFilterType::New();
  m_ZeroMultiplyFilter2 = MultiplyFilterType::New();

  // Set constant parameters
  m_ZeroMultiplyFilter->SetConstant2(itk::NumericTraits<typename ProjectionStackType::PixelType>::ZeroValue());
  m_ZeroMultiplyFilter2->SetConstant2(itk::NumericTraits<typename ProjectionStackType::PixelType>::ZeroValue());

  // Set permanent connections
  m_ExtractFilter->SetInput(m_ZeroMultiplyFilter->GetOutput());

  // Set memory management flags
//  m_ZeroMultiplyFilter->ReleaseDataFlagOn();
  m_InterpolationFilter->ReleaseDataFlagOn();
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetInputProjectionStack(const ProjectionStackType* Projection)
{
  this->SetNthInput(0, const_cast<ProjectionStackType*>(Projection));
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetNthInput(1, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename ProjectionStackType, typename VolumeSeriesType>
typename ProjectionStackType::Pointer
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GetInputProjectionStack()
{
  return static_cast< ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template< typename ProjectionStackType, typename VolumeSeriesType>
typename VolumeSeriesType::ConstPointer
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg)
{
  m_ForwardProjectionFilter = _arg;
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetWeights(const itk::Array2D<float> _arg)
{
  m_Weights = _arg;
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::SetGeometry(GeometryType::Pointer _arg)
{
  m_Geometry=_arg;
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::InitializeConstantSource()
{
  int Dimension = 3;

  typename ProjectionStackType::SizeType constantImageSourceSize;
  constantImageSourceSize.Fill(0);
  for(unsigned int i=0; i < Dimension; i++)
      constantImageSourceSize[i] = GetInputVolumeSeries()->GetLargestPossibleRegion().GetSize()[i];

  typename ProjectionStackType::SpacingType constantImageSourceSpacing;
  constantImageSourceSpacing.Fill(0);
  for(unsigned int i=0; i < Dimension; i++)
      constantImageSourceSpacing[i] = GetInputVolumeSeries()->GetSpacing()[i];

  typename ProjectionStackType::PointType constantImageSourceOrigin;
  constantImageSourceOrigin.Fill(0);
  for(unsigned int i=0; i < Dimension; i++)
      constantImageSourceOrigin[i] = GetInputVolumeSeries()->GetOrigin()[i];

  typename ProjectionStackType::DirectionType constantImageSourceDirection;
  constantImageSourceDirection.SetIdentity();

  m_ConstantSource->SetOrigin( constantImageSourceOrigin );
  m_ConstantSource->SetSpacing( constantImageSourceSpacing );
  m_ConstantSource->SetDirection( constantImageSourceDirection );
  m_ConstantSource->SetSize( constantImageSourceSize );
  m_ConstantSource->SetConstant( 0. );
  m_ConstantSource->Update();
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GenerateOutputInformation()
{
  // Connect the filters
  m_ZeroMultiplyFilter->SetInput1(this->GetInputProjectionStack());
  m_ZeroMultiplyFilter2->SetInput1(this->GetInputProjectionStack());
  m_InterpolationFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
  m_InterpolationFilter->SetInputVolume(m_ConstantSource->GetOutput());
//  m_PasteFilter->SetDestinationImage(this->GetInputProjectionStack());
  m_PasteFilter->SetDestinationImage(m_ZeroMultiplyFilter2->GetOutput());

  // Connections with the Forward projection filter can only be set at runtime
  m_ForwardProjectionFilter->SetInput(0, m_ExtractFilter->GetOutput());
  m_ForwardProjectionFilter->SetInput(1, m_InterpolationFilter->GetOutput());
  m_PasteFilter->SetSourceImage(m_ForwardProjectionFilter->GetOutput());

  // Set runtime parameters
  int Dimension = ProjectionStackType::ImageDimension; // Dimension = 3
  m_InterpolationFilter->SetWeights(m_Weights);
  m_ForwardProjectionFilter->SetGeometry(m_Geometry);

  typename ProjectionStackType::RegionType extractRegion;
  extractRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  extractRegion.SetSize(Dimension-1, 1);
  extractRegion.SetIndex(Dimension-1, m_ProjectionNumber);
  m_ExtractFilter->SetExtractionRegion(extractRegion);

  typename ProjectionStackType::RegionType PasteSourceRegion;
  PasteSourceRegion.SetSize(extractRegion.GetSize());
  PasteSourceRegion.SetIndex(extractRegion.GetIndex());
  PasteSourceRegion.SetIndex(Dimension-1, 0);
  m_PasteFilter->SetSourceRegion(PasteSourceRegion);
  m_PasteFilter->SetDestinationIndex(extractRegion.GetIndex());

  // Have the last filter calculate its output information
  this->InitializeConstantSource();
  m_PasteFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation(m_PasteFilter->GetOutput());
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GenerateInputRequestedRegion()
{
  // Input 0 is the stack of projections
  typename ProjectionStackType::Pointer  inputPtr0 = const_cast< ProjectionStackType * >( this->GetInput(0) );
  if ( !inputPtr0 )
    {
    return;
    }
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the volume series we update
  typename VolumeSeriesType::Pointer inputPtr1 = static_cast< VolumeSeriesType * >
            ( this->itk::ProcessObject::GetInput(1) );
  inputPtr1->SetRequestedRegionToLargestPossibleRegion();
}

template< typename ProjectionStackType, typename VolumeSeriesType>
void
FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
::GenerateData()
{
  int Dimension = ProjectionStackType::ImageDimension;

  // Set the Extract filter
  typename ProjectionStackType::RegionType extractRegion;
  extractRegion = this->GetInputProjectionStack()->GetLargestPossibleRegion();
  extractRegion.SetSize(Dimension-1, 1);

  int NumberProjs = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(2);
  for(m_ProjectionNumber=0; m_ProjectionNumber<NumberProjs; m_ProjectionNumber++)
    {

    // After the first update, we need to use the output as input.
    if(m_ProjectionNumber>0)
      {
      typename ProjectionStackType::Pointer pimg = m_PasteFilter->GetOutput();
      pimg->DisconnectPipeline();
      m_PasteFilter->SetDestinationImage( pimg );
      }

    // Set the Extract Filter
    extractRegion.SetIndex(Dimension-1, m_ProjectionNumber);
    m_ExtractFilter->SetExtractionRegion(extractRegion);

    // Set the Paste Filter
    m_PasteFilter->SetSourceRegion(extractRegion);
    m_PasteFilter->SetDestinationIndex(extractRegion.GetIndex());

    // Set the Interpolation filter
    m_InterpolationFilter->SetProjectionNumber(m_ProjectionNumber);

    // Update the last filter
    m_PasteFilter->Update();
    }

  // Graft its output
  this->GraftOutput( m_PasteFilter->GetOutput() );
}

}// end namespace


#endif
