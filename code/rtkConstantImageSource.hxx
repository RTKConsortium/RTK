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

#ifndef rtkConstantImageSource_hxx
#define rtkConstantImageSource_hxx

#include <itkImageRegionIterator.h>

namespace rtk
{

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
    m_Index[i] = 0;

    for (unsigned int j=0; j<TOutputImage::GetImageDimension(); j++)
      m_Direction[i][j] = (i==j)?1.:0.;
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
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "Constant: "
     << static_cast<typename itk::NumericTraits<OutputImagePixelType>::PrintType>(m_Constant)
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

template <class TOutputImage>
void
ConstantImageSource<TOutputImage>
::SetInformationFromImage(const typename TOutputImage::Superclass* image)
{
  this->SetSize( image->GetLargestPossibleRegion().GetSize() );
  this->SetIndex( image->GetLargestPossibleRegion().GetIndex() );
  this->SetSpacing( image->GetSpacing() );
  this->SetOrigin( image->GetOrigin() );
  this->SetDirection( image->GetDirection() );
}

//----------------------------------------------------------------------------
template <typename TOutputImage>
void 
ConstantImageSource<TOutputImage>
::GenerateOutputInformation()
{
  TOutputImage *output;
  output = this->GetOutput(0);

  typename TOutputImage::RegionType largestPossibleRegion;
  largestPossibleRegion.SetSize( this->m_Size );
  largestPossibleRegion.SetIndex( this->m_Index );
  output->SetLargestPossibleRegion( largestPossibleRegion );

  output->SetSpacing(m_Spacing);
  output->SetOrigin(m_Origin);
  output->SetDirection(m_Direction);
}

//----------------------------------------------------------------------------
template <typename TOutputImage>
void 
ConstantImageSource<TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  itk::ImageRegionIterator<TOutputImage> it(this->GetOutput(), outputRegionForThread);
  for (; !it.IsAtEnd(); ++it)
    it.Set( this->GetConstant() );
}

} // end namespace rtk

#endif
