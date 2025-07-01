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

#ifndef rtkElektaSynergyRawLookupTableImageFilter_hxx
#define rtkElektaSynergyRawLookupTableImageFilter_hxx


namespace rtk
{

template <class TInputImage, class TOutputImage>
ElektaSynergyRawLookupTableImageFilter<TInputImage, TOutputImage>::ElektaSynergyRawLookupTableImageFilter()
{
  // Create the lut
  auto                               lut = LookupTableType::New();
  typename LookupTableType::SizeType size;
  size[0] = itk::NumericTraits<InputImagePixelType>::max() - itk::NumericTraits<InputImagePixelType>::min() + 1;
  lut->SetRegions(size);
  lut->Allocate();

  // Iterate and set lut
  itk::ImageRegionIteratorWithIndex<LookupTableType> it(lut, lut->GetBufferedRegion());
  it.GoToBegin();

  // First value takes value of pixel #1
  it.Set(OutputImagePixelType(size[0] - 1));
  ++it;

  // Conventional lookup table for the rest
  while (!it.IsAtEnd())
  {
    it.Set(OutputImagePixelType(size[0] - it.GetIndex()[0]));
    ++it;
  }

  // Last value takes value of pixel #1
  --it;
  it.Set(OutputImagePixelType(size[0] - 1));

  // Set the lut to member and functor
  this->SetLookupTable(lut);
}

} // namespace rtk

#endif
