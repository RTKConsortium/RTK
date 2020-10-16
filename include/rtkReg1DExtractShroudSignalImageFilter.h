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

#ifndef rtkReg1DExtractShroudSignalImageFilter_h
#define rtkReg1DExtractShroudSignalImageFilter_h

#include <itkImageToImageFilter.h>

namespace rtk
{
/** \class Reg1DExtractShroudSignalImageFilter
 * \brief Reg1DExtract the signal corresponding to the breathing motion
 * (1D) from a shroud image (2D).
 *
 * \test rtkamsterdamshroudtest.cxx
 *
 * \author Vivien Delmon
 *
 * \ingroup RTK ImageToImageFilter
 */

template <class TInputPixel, class TOutputPixel>
class ITK_EXPORT Reg1DExtractShroudSignalImageFilter
  : public itk::ImageToImageFilter<itk::Image<TInputPixel, 2>, itk::Image<TOutputPixel, 1>>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(Reg1DExtractShroudSignalImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(Reg1DExtractShroudSignalImageFilter);
#endif

  /** Standard class type alias. */
  using TInputImage = itk::Image<TInputPixel, 2>;
  using TOutputImage = itk::Image<TOutputPixel, 1>;
  using Self = Reg1DExtractShroudSignalImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);
  itkStaticConstMacro(ImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(Reg1DExtractShroudSignalImageFilter, itk::ImageToImageFilter);

protected:
  Reg1DExtractShroudSignalImageFilter();
  ~Reg1DExtractShroudSignalImageFilter() override = default;

  void
  GenerateOutputInformation() override;
  void
  GenerateInputRequestedRegion() override;
  void
  GenerateData() override;

private:
  using RegisterImageType = itk::Image<TInputPixel, 1>;
  TOutputPixel
  register1D(const RegisterImageType *, const RegisterImageType *);

}; // end of class

} // end of namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkReg1DExtractShroudSignalImageFilter.hxx"
#endif

#endif // ! rtkReg1DExtractShroudSignalImageFilter_h
