#ifndef __itkConstantImageSource_txx
#define __itkConstantImageSource_txx

#include "itkConstantImageSource.h"
#include "itkImageRegionIterator.h"
 
namespace itk
{

/**
 *
 */
template <class TOutputImage>
ConstantImageSource<TOutputImage>
::ConstantImageSource()
{
  //Initial image is 64 wide in each direction.
  for (unsigned int i=0; i<TOutputImage::GetImageDimension(); i++)
    {
    m_Size[i] = 64;
    m_Spacing[i] = 1.0;
    m_Origin[i] = 0.0;
    }

  m_Constant = 0.;
}

template <class TOutputImage>
ConstantImageSource<TOutputImage>
::~ConstantImageSource()
{
}

template <class TOutputImage>
void
ConstantImageSource<TOutputImage>
::SetSize( SizeValueArrayType sizeArray )
{
  const unsigned int count = TOutputImage::ImageDimension;
  unsigned int i;
  for( i=0; i<count; i++ )
    {
    if( sizeArray[i] != this->m_Size[i] )
      {
      break;
      }
    }
  if( i < count )
    {
    this->Modified();
    for( i=0; i<count; i++ )
      {
      this->m_Size[i] = sizeArray[i];
      }
    }
}

template <class TOutputImage>
const typename ConstantImageSource<TOutputImage>::SizeValueType *
ConstantImageSource<TOutputImage>
::GetSize() const
{
  return this->m_Size.GetSize();
}

template <class TOutputImage>
void
ConstantImageSource<TOutputImage>
::SetSpacing( SpacingValueArrayType spacingArray )
{
  const unsigned int count = TOutputImage::ImageDimension;
  unsigned int i;
  for( i=0; i<count; i++ )
    {
    if( spacingArray[i] != this->m_Spacing[i] )
      {
      break;
      }
    }
  if( i < count )
    {
    this->Modified();
    for( i=0; i<count; i++ )
      {
      this->m_Spacing[i] = spacingArray[i];
      }
    }
}

template <class TOutputImage>
void
ConstantImageSource<TOutputImage>
::SetOrigin( PointValueArrayType originArray )
{
  const unsigned int count = TOutputImage::ImageDimension;
  unsigned int i;
  for( i=0; i<count; i++ )
    {
    if( originArray[i] != this->m_Origin[i] )
      {
      break;
      }
    }
  if( i < count )
    {
    this->Modified();
    for( i=0; i<count; i++ )
      {
      this->m_Origin[i] = originArray[i];
      }
    }
}

template <class TOutputImage>
const typename ConstantImageSource<TOutputImage>::PointValueType *
ConstantImageSource<TOutputImage>
::GetOrigin() const
{
  for(unsigned int i=0; i < TOutputImage::ImageDimension; i++ )
    {
    this->m_OriginArray[i] = this->m_Origin[i];
    }
  return this->m_OriginArray;
}

template <class TOutputImage>
const typename ConstantImageSource<TOutputImage>::SpacingValueType *
ConstantImageSource<TOutputImage>
::GetSpacing() const
{
  for(unsigned int i=0; i < TOutputImage::ImageDimension; i++ )
    {
    this->m_SpacingArray[i] = this->m_Spacing[i];
    }
  return this->m_SpacingArray;
}


/**
 *
 */
template <class TOutputImage>
void 
ConstantImageSource<TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "Constant: "
     << static_cast<typename NumericTraits<OutputImagePixelType>::PrintType>(m_Constant)
     << std::endl;
  unsigned int i;
  os << indent << "Origin: [";
  for (i=0; i < TOutputImage::ImageDimension - 1; i++)
    {
    os << m_Origin[i] << ", ";
    }
  os << m_Origin[i] << "]" << std::endl;

  os << indent << "Spacing: [";
  for (i=0; i < TOutputImage::ImageDimension - 1; i++)
    {
    os << m_Spacing[i] << ", ";
    }
  os << m_Spacing[i] << "]" << std::endl;

  os << indent << "Size: [";
  for (i=0; i < TOutputImage::ImageDimension - 1; i++)
    {
    os << m_Size[i] << ", ";
    }
  os << m_Size[i] << "]" << std::endl;
}

//----------------------------------------------------------------------------
template <typename TOutputImage>
void 
ConstantImageSource<TOutputImage>
::GenerateOutputInformation()
{
  TOutputImage *output;
  IndexType index;
  index.Fill( 0 );
  
  output = this->GetOutput(0);

  typename TOutputImage::RegionType largestPossibleRegion;
  largestPossibleRegion.SetSize( this->m_Size );
  largestPossibleRegion.SetIndex( index );
  output->SetLargestPossibleRegion( largestPossibleRegion );

  output->SetSpacing(m_Spacing);
  output->SetOrigin(m_Origin);
}

//----------------------------------------------------------------------------
template <typename TOutputImage>
void 
ConstantImageSource<TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId )
{
  ImageRegionIterator<TOutputImage> it(this->GetOutput(), outputRegionForThread);
  for (; !it.IsAtEnd(); ++it)
    it.Set( this->GetConstant() );
}

} // end namespace itk

#endif
