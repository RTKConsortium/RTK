#ifndef __rtkProjectionStackToFourDImageFilter_txx
#define __rtkProjectionStackToFourDImageFilter_txx

#include "rtkProjectionStackToFourDImageFilter.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

namespace rtk
{

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::ProjectionStackToFourDImageFilter()
{
  this->SetNumberOfRequiredInputs(2);

  m_ProjectionNumber = 0;
  m_UseCudaSplat = false;

  // Create the filters
  m_ExtractFilter = ExtractFilterType::New();
  m_ConstantImageSource = ConstantImageSourceType::New();

  m_ZeroMultiplyFilter = MultiplyFilterType::New();

  // Set constant parameters
  m_ZeroMultiplyFilter->SetConstant2(itk::NumericTraits<typename VolumeSeriesType::PixelType>::ZeroValue());
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::SetInputProjectionStack(const ProjectionStackType* Projection)
{
  this->SetNthInput(1, const_cast<ProjectionStackType*>(Projection));
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
typename VolumeSeriesType::ConstPointer ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
typename ProjectionStackType::ConstPointer ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>::GetInputProjectionStack()
{
  return static_cast< const ProjectionStackType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg)
{
  m_BackProjectionFilter = _arg;
  this->Modified();
}

//template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
//void
//ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
//::SetWeights(const itk::Array2D<float> _arg)
//{
//  m_Weights = _arg;
//  this->Modified();
//}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg)
{
  m_Geometry = _arg;
  this->Modified();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::InitializeConstantSource()
{
  int Dimension = 3;

  // Configure the constant image source that is connected to input 2 of the m_SingleProjToFourDFilter
  typename VolumeType::SizeType constantImageSourceSize;
  constantImageSourceSize.Fill(0);
  for(unsigned int i=0; i < Dimension; i++)
      constantImageSourceSize[i] = GetInputVolumeSeries()->GetLargestPossibleRegion().GetSize()[i];

  typename VolumeType::SpacingType constantImageSourceSpacing;
  constantImageSourceSpacing.Fill(0);
  for(unsigned int i=0; i < Dimension; i++)
      constantImageSourceSpacing[i] = GetInputVolumeSeries()->GetSpacing()[i];

  typename VolumeType::PointType constantImageSourceOrigin;
  constantImageSourceOrigin.Fill(0);
  for(unsigned int i=0; i < Dimension; i++)
      constantImageSourceOrigin[i] = constantImageSourceSpacing[i] * (constantImageSourceSize[i]-1) * -0.5;

  typename VolumeType::DirectionType constantImageSourceDirection;
  constantImageSourceDirection.SetIdentity();

  m_ConstantImageSource->SetOrigin( constantImageSourceOrigin );
  m_ConstantImageSource->SetSpacing( constantImageSourceSpacing );
  m_ConstantImageSource->SetDirection( constantImageSourceDirection );
  m_ConstantImageSource->SetSize( constantImageSourceSize );
  m_ConstantImageSource->SetConstant( 0. );
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::GenerateOutputInformation()
{
  // Set runtime connections
  m_ExtractFilter->SetInput(this->GetInputProjectionStack());
  m_ZeroMultiplyFilter->SetInput1(this->GetInputVolumeSeries());

  m_BackProjectionFilter->SetInput(0, m_ConstantImageSource->GetOutput());
  m_BackProjectionFilter->SetInput(1, m_ExtractFilter->GetOutput());

  // Create and set the splat filter
  if (m_UseCudaSplat) m_SplatFilter = rtk::CudaSplatImageFilter::New();
      else m_SplatFilter = SplatFilterType::New();
  m_SplatFilter->SetInputVolumeSeries(m_ZeroMultiplyFilter->GetOutput());
  m_SplatFilter->SetInputVolume(m_BackProjectionFilter->GetOutput());

  // Set runtime parameters
  m_BackProjectionFilter->SetGeometry(m_Geometry.GetPointer());
  m_SplatFilter->SetProjectionNumber(m_ProjectionNumber);
  m_SplatFilter->SetWeights(m_Weights);

  // Prepare the extract filter
  int Dimension = ProjectionStackType::ImageDimension; // Dimension=3
  int NumberProjs = GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(2);
  if (NumberProjs != m_Weights.columns()){
      std::cerr << "Size of interpolation weights array does not match the number of projections" << std::endl;
  }

  typename ExtractFilterType::InputImageRegionType subsetRegion;
  subsetRegion = GetInputProjectionStack()->GetLargestPossibleRegion();
  subsetRegion.SetSize(Dimension-1, 1);
  subsetRegion.SetIndex(Dimension-1, m_ProjectionNumber);
  m_ExtractFilter->SetExtractionRegion(subsetRegion);

  // Have the last filter calculate its output information
  this->InitializeConstantSource();
  m_SplatFilter->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation(m_SplatFilter->GetOutput());
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
::GenerateInputRequestedRegion()
{
  // Input 0 is the stack of projections
  typename VolumeSeriesType::Pointer  inputPtr0 = const_cast< VolumeSeriesType * >( this->GetInput(0) );
  if ( !inputPtr0 )
    {
    return;
    }
  inputPtr0->SetRequestedRegion( this->GetOutput()->GetRequestedRegion() );

  // Input 1 is the volume series we update
  typename ProjectionStackType::Pointer inputPtr1 = static_cast< ProjectionStackType * >
            ( this->itk::ProcessObject::GetInput(1) );
  inputPtr1->SetRequestedRegionToLargestPossibleRegion();
}

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType, TFFTPrecision>
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
      typename VolumeSeriesType::Pointer pimg = m_SplatFilter->GetOutput();
      pimg->DisconnectPipeline();
      m_SplatFilter->SetInputVolumeSeries( pimg );
      }

    // Set the Extract Filter
    extractRegion.SetIndex(Dimension-1, m_ProjectionNumber);
    m_ExtractFilter->SetExtractionRegion(extractRegion);

    // Set the splat filter
    m_SplatFilter->SetProjectionNumber(m_ProjectionNumber);

    // Update the last filter
    m_SplatFilter->Update();
    }

  // Graft its output
  this->GraftOutput( m_SplatFilter->GetOutput() );
}

}// end namespace


#endif
