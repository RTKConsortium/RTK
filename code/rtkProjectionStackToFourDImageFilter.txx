#ifndef __rtkProjectionStackToFourDImageFilter_txx
#define __rtkProjectionStackToFourDImageFilter_txx

#include "rtkProjectionStackToFourDImageFilter.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

#include "rtkJosephBackProjectionImageFilter.h"

namespace rtk
{

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>::ProjectionStackToFourDImageFilter()
{
    this->SetNumberOfRequiredInputs(2);

    // Default behaviour is unfiltered backprojection
    m_UseRampFilter=false;

    // Create the two filters
    m_SingleProjToFourDFilter = SingleProjectionToFourDFilterType::New();
    m_ExtractFilter = ExtractFilterType::New();
    m_constantImageSource = ConstantImageSourceType::New();

    // Connect them
    m_SingleProjToFourDFilter->SetInputProjectionStack(m_ExtractFilter->GetOutput());
    m_SingleProjToFourDFilter->SetInputEmptyVolume(m_constantImageSource->GetOutput());
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
void ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
    this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
void ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>::SetInputProjectionStack(const VolumeType* Projection)
{
    this->SetNthInput(1, const_cast<VolumeType*>(Projection));
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
typename VolumeSeriesType::ConstPointer ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>::GetInputVolumeSeries()
{
    return static_cast< const VolumeSeriesType * >
            ( this->itk::ProcessObject::GetInput(0) );
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
typename VolumeType::ConstPointer ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>::GetInputProjectionStack()
{
    return static_cast< const VolumeType * >
            ( this->itk::ProcessObject::GetInput(1) );
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>
::SetBackProjectionFilter (const BackProjectionFilterPointer _arg)
{
    m_SingleProjToFourDFilter->SetBackProjectionFilter(_arg);
    this->Modified();
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>
::SetWeights(const itk::Array2D<float> _arg)
{
    m_SingleProjToFourDFilter->SetWeights(_arg);
    m_Weights = _arg;
    this->Modified();
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>
::SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg)
{
    m_SingleProjToFourDFilter->SetGeometry(_arg);
    m_Geometry = _arg;
    this->Modified();
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>
::SetUseRampFilter(bool arg)
{
    m_UseRampFilter=arg;
    this->Modified();
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>
::SetUseCuda(bool arg)
{
    m_SingleProjToFourDFilter->SetUseCuda(arg);
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>
::GenerateOutputInformation()
{
  if (m_UseRampFilter)
  {
      typename RampFilterType::Pointer RampFilter = RampFilterType::New();
      RampFilter->SetInput(this->GetInputProjectionStack());
      m_ExtractFilter->SetInput(RampFilter->GetOutput());
  }
  else
  {
      m_ExtractFilter->SetInput(this->GetInputProjectionStack());
  }
  m_SingleProjToFourDFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());

  int NumberProjs = GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(2);
  if (NumberProjs != m_SingleProjToFourDFilter->GetWeights().columns()){
      std::cerr << "Size of interpolation weights array does not match the number of projections" << std::endl;
  }

  int Dimension = this->GetInputVolumeSeries()->GetImageDimension();

  // Configure the constant image source that is connected to input 2 of the m_SingleProjToFourDFilter
  typename VolumeType::SizeType constantImageSourceSize;
  constantImageSourceSize.Fill(0);
  for(unsigned int i=0; i < Dimension - 1; i++)
      constantImageSourceSize[i] = GetInputVolumeSeries()->GetLargestPossibleRegion().GetSize()[i];

  typename VolumeType::SpacingType constantImageSourceSpacing;
  constantImageSourceSpacing.Fill(0);
  for(unsigned int i=0; i < Dimension - 1; i++)
      constantImageSourceSpacing[i] = GetInputVolumeSeries()->GetSpacing()[i];

  typename VolumeType::PointType constantImageSourceOrigin;
  constantImageSourceOrigin.Fill(0);
  for(unsigned int i=0; i < Dimension - 1; i++)
      constantImageSourceOrigin[i] = constantImageSourceSpacing[i] * (constantImageSourceSize[i]-1) * -0.5;

  typename VolumeType::DirectionType constantImageSourceDirection;
  constantImageSourceDirection.SetIdentity();

  m_constantImageSource->SetOrigin( constantImageSourceOrigin );
  m_constantImageSource->SetSpacing( constantImageSourceSpacing );
  m_constantImageSource->SetDirection( constantImageSourceDirection );
  m_constantImageSource->SetSize( constantImageSourceSize );
  m_constantImageSource->SetConstant( 0. );

  typename ExtractFilterType::InputImageRegionType subsetRegion;
  typename ExtractFilterType::InputImageSizeType subsetSize;
  typename ExtractFilterType::InputImageIndexType subsetIndex;

  subsetRegion = GetInputProjectionStack()->GetLargestPossibleRegion();
  subsetSize = subsetRegion.GetSize();
  subsetIndex = subsetRegion.GetIndex();

  subsetSize[2] = 1;
  subsetIndex[2] = 0;

  subsetRegion.SetSize(subsetSize);
  subsetRegion.SetIndex(subsetIndex);
  m_ExtractFilter->SetExtractionRegion(subsetRegion);

  // Have the last filter calculate its output information
  std::cout << "In ProjectionStackToFourDImageFilter. About to UpdateOutputInformation" << std::endl;
  m_ExtractFilter->UpdateOutputInformation();
  m_SingleProjToFourDFilter->UpdateOutputInformation();
  std::cout << "In ProjectionStackToFourDImageFilter. UpdateOutputInformation complete" << std::endl;

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation(m_SingleProjToFourDFilter->GetOutput());
}

template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision>
void
ProjectionStackToFourDImageFilter<VolumeSeriesType, VolumeType, TFFTPrecision>
::GenerateData()
{
  std::cout << "In ProjectionStackToFourDImageFilter : Entering GenerateData()" << std::endl;
  int NumberProjs = GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(2);
  if (NumberProjs != m_SingleProjToFourDFilter->GetWeights().columns()){
      std::cerr << "Size of interpolation weights array does not match the number of projections" << std::endl;
  }

    typename ExtractFilterType::InputImageRegionType subsetRegion;
    typename ExtractFilterType::InputImageSizeType subsetSize;
    typename ExtractFilterType::InputImageIndexType subsetIndex;

    for(unsigned int i=0; i<NumberProjs; i++)
    {
        // After the first update, we need to use its output as input
        if(i>0)
        {
            typename VolumeSeriesType::Pointer pimg = m_SingleProjToFourDFilter->GetOutput();
            pimg->DisconnectPipeline();
            m_SingleProjToFourDFilter->SetInput(0, pimg );
        }

        // The backprojection works on one projection at a time : set extract filter to select a single projection

        subsetRegion = GetInputProjectionStack()->GetLargestPossibleRegion();
        subsetSize = subsetRegion.GetSize();
        subsetIndex = subsetRegion.GetIndex();

        subsetSize[2] = 1;
        subsetIndex[2] = i;

        subsetRegion.SetSize(subsetSize);
        subsetRegion.SetIndex(subsetIndex);
        m_ExtractFilter->SetExtractionRegion(subsetRegion);

        m_SingleProjToFourDFilter->Update();

    }

    this->GraftOutput( m_SingleProjToFourDFilter->GetOutput() );
    std::cout << "In ProjectionStackToFourDImageFilter : Leaving GenerateData()" << std::endl;
}

}// end namespace


#endif
