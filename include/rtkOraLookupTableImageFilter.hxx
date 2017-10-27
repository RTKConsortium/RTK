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

#ifndef rtkOraLookupTableImageFilter_hxx
#define rtkOraLookupTableImageFilter_hxx

#include <itkImageIOBase.h>

namespace rtk
{

template <class TOutputImage>
void
OraLookupTableImageFilter<TOutputImage>
::BeforeThreadedGenerateData()
{
  // Create the lut
  typename LookupTableType::Pointer lut = LookupTableType::New();
  typename LookupTableType::SizeType size;
  size[0] = itk::NumericTraits<InputImagePixelType>::max() -
            itk::NumericTraits<InputImagePixelType>::min() + 1;
  lut->SetRegions( size );
  lut->Allocate();

  // Read meta data dictionary from file
  int fileIdx = this->GetOutput()->GetRequestedRegion().GetIndex()[2];
  itk::ImageIOBase::Pointer reader;
  reader = itk::ImageIOFactory::CreateImageIO(m_FileNames[fileIdx].c_str(),
                                              itk::ImageIOFactory::ReadMode);
  if (!reader)
    {
    itkExceptionMacro("Error reading file " << m_FileNames[fileIdx]);
    }
  reader->SetFileName( m_FileNames[fileIdx].c_str() );
  reader->ReadImageInformation();
  const itk::MetaDataDictionary &dic = reader->GetMetaDataDictionary();

  // Retrieve and set slope / intercept
  double slope = 1.;
  typedef itk::MetaDataObject< double > MetaDataDoubleType;
  const MetaDataDoubleType *slopeMeta = dynamic_cast<const MetaDataDoubleType *>( dic["rescale_slope"]);
  if(slopeMeta!=ITK_NULLPTR)
    {
    slope = slopeMeta->GetMetaDataObjectValue();
    }

  double intercept = 0.;
  const MetaDataDoubleType *interceptMeta = dynamic_cast<const MetaDataDoubleType *>( dic["rescale_intercept"] );
  if(interceptMeta!=ITK_NULLPTR)
    {
    intercept = interceptMeta->GetMetaDataObjectValue();
    }

  // Iterate and set lut
  itk::ImageRegionIteratorWithIndex<LookupTableType> it( lut, lut->GetBufferedRegion() );
  it.GoToBegin();
  if(m_ComputeLineIntegral)
    {
    int negidx = itk::Math::Floor<int, double>(std::floor(-intercept/slope));
    double negval = -1. * vcl_log(slope * (negidx+1) + intercept);
    while( !it.IsAtEnd() && (int)it.GetIndex()[0] <= negidx )
      {
      it.Set(negval);
      ++it;
      }
    while( !it.IsAtEnd() )
      {
      it.Set( -1. * vcl_log(slope * it.GetIndex()[0] + intercept) );
      ++it;
      }
    }
  else
    {
    while( !it.IsAtEnd() )
      {
      it.Set( slope * it.GetIndex()[0] + intercept );
      ++it;
      }
    }
  this->SetLookupTable( lut );
  Superclass::BeforeThreadedGenerateData(); // Update the LUT
}

}

#endif
