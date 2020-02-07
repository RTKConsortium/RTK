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

#ifndef rtkReconstructImageFilter_h
#define rtkReconstructImageFilter_h

// Includes
#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkNaryAddImageFilter.h>

#include "rtkDaubechiesWaveletsConvolutionImageFilter.h"
#include "rtkUpsampleImageFilter.h"

namespace rtk
{

/**
 * \class ReconstructImageFilter
 * \brief An image filter that reconstructs an image using
 * Daubechies wavelets.
 *
 * This filter is inspired from Dan Mueller's GIFT package
 * http://www.insight-journal.org/browse/publication/103
 *
 * \dot
 * digraph ReconstructImageFilter {
 *
 * Output [ label="Output"];
 * Output [shape=Mdiamond];
 * Input0 [label="Input 0 (here, 2D)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (here, 2D)"];
 * Input1 [shape=Mdiamond];
 * Input2 [label="Input 2 (here, 2D)"];
 * Input2 [shape=Mdiamond];
 * Input3 [label="Input 3 (here, 2D)"];
 * Input3 [shape=Mdiamond];
 * Input4 [label="Input 4 (here, 2D)"];
 * Input4 [shape=Mdiamond];
 * Input5 [label="Input 5 (here, 2D)"];
 * Input5 [shape=Mdiamond];
 * Input6 [label="Input 6 (here, 2D)"];
 * Input6 [shape=Mdiamond];
 *
 * node [shape=box];
 * Add0 [ label="itk::NaryAddImageFilter" URL="\ref itk::NaryAddImageFilter"];
 * Add1 [ label="itk::NaryAddImageFilter" URL="\ref itk::NaryAddImageFilter"];
 * Conv0 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Lowpass, Lowpass)"
 *         URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv1 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Lowpass, Highpass)"
 *         URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv2 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Highpass, Lowpass)"
 *         URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv3 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Highpass, Highpass)"
 *         URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv4 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Lowpass, Lowpass)"
 *         URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv5 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Lowpass, Highpass)"
 *         URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv6 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Highpass, Lowpass)"
 *         URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv7 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Highpass, Highpass)"
 *         URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Up0 [ label="rtk::UpsampleImageFilter (by 2)" URL="\ref rtk::UpsampleImageFilter"];
 * Up1 [ label="rtk::UpsampleImageFilter (by 2)" URL="\ref rtk::UpsampleImageFilter"];
 * Up2 [ label="rtk::UpsampleImageFilter (by 2)" URL="\ref rtk::UpsampleImageFilter"];
 * Up3 [ label="rtk::UpsampleImageFilter (by 2)" URL="\ref rtk::UpsampleImageFilter"];
 * Up4 [ label="rtk::UpsampleImageFilter (by 2)" URL="\ref rtk::UpsampleImageFilter"];
 * Up5 [ label="rtk::UpsampleImageFilter (by 2)" URL="\ref rtk::UpsampleImageFilter"];
 * Up6 [ label="rtk::UpsampleImageFilter (by 2)" URL="\ref rtk::UpsampleImageFilter"];
 * Up7 [ label="rtk::UpsampleImageFilter (by 2)" URL="\ref rtk::UpsampleImageFilter"];
 * Input0 -> Up0;
 * Input1 -> Up1;
 * Input2 -> Up2;
 * Input3 -> Up3;
 * Input4 -> Up5;
 * Input5 -> Up6;
 * Input6 -> Up7;
 * Up0 -> Conv0;
 * Up1 -> Conv1;
 * Up2 -> Conv2;
 * Up3 -> Conv3;
 * Up4 -> Conv4;
 * Up5 -> Conv5;
 * Up6 -> Conv6;
 * Up7 -> Conv7;
 * Conv0 -> Add0;
 * Conv1 -> Add0;
 * Conv2 -> Add0;
 * Conv3 -> Add0;
 * Conv4 -> Add1;
 * Conv5 -> Add1;
 * Conv6 -> Add1;
 * Conv7 -> Add1;
 * Add0 -> Up4;
 * Add1 -> Output;
 * }
 * \enddot
 *
 * \test rtkwaveletstest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <class TImage>
class ReconstructImageFilter : public itk::ImageToImageFilter<TImage, TImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(ReconstructImageFilter);

  /** Standard class type alias. */
  using Self = ReconstructImageFilter;
  using Superclass = itk::ImageToImageFilter<TImage, TImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ReconstructImageFilter, ImageToImageFilter);

  /** ImageDimension enumeration. */
  static constexpr unsigned int ImageDimension = TImage::ImageDimension;

  /** Inherit types from Superclass. */
  using InputImageType = typename Superclass::InputImageType;
  using OutputImageType = typename Superclass::OutputImageType;
  using InputImagePointer = typename Superclass::InputImagePointer;
  using OutputImagePointer = typename Superclass::OutputImagePointer;
  using InputImageConstPointer = typename Superclass::InputImageConstPointer;
  using PixelType = typename TImage::PixelType;
  using InternalPixelType = typename TImage::InternalPixelType;

  /** Typedefs for pipeline's subfilters */
  using AddFilterType = itk::NaryAddImageFilter<InputImageType, InputImageType>;
  using ConvolutionFilterType = rtk::DaubechiesWaveletsConvolutionImageFilter<InputImageType>;
  using UpsampleImageFilterType = rtk::UpsampleImageFilter<InputImageType>;

  /** Set the number of input levels. */
  virtual void
  SetNumberOfLevels(unsigned int levels)
  {
    this->m_NumberOfLevels = levels;
    this->ModifyInputOutputStorage();
  }

  /** Get the number of input levels (per image). */
  virtual unsigned int
  GetNumberOfLevels()
  {
    return this->m_NumberOfLevels;
  }

  /** ReconstructImageFilter produces images which are of different size
   *  than the input image. As such, we reimplement GenerateOutputInformation()
   *  in order to inform the pipeline execution model.
   */
  void
  GenerateOutputInformation() override;


  /** ReconstructImageFilter requests the largest possible region of all its inputs.
   */
  void
  GenerateInputRequestedRegion() override;

  /** ReconstructImageFilter uses input images of different sizes, therefore the
   * VerifyInputInformation method has to be reimplemented.
   */
  void
  VerifyInputInformation() const override
  {}

  void
  SetSizes(typename InputImageType::SizeType * sizesVector)
  {
    m_Sizes = sizesVector;
  }

  void
  SetIndices(typename InputImageType::IndexType * indicesVector)
  {
    m_Indices = indicesVector;
  }

  /** Get/Set the order of the wavelet filters */
  itkGetMacro(Order, unsigned int);
  itkSetMacro(Order, unsigned int);

protected:
  ReconstructImageFilter();
  ~ReconstructImageFilter() override = default;

  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

  /** Modifies the storage for Input and Output images.
   * Should be called after changes to levels, bands,
   * Reconstruct, reconstruct, etc... */
  void
  ModifyInputOutputStorage();

  /** Does the real work. */
  void
  GenerateData() override;

  /** Calculates the number of ProcessObject output images */
  virtual unsigned int
  CalculateNumberOfInputs();

  /** Creates and sets the kernel sources to generate all kernels. */
  void
  GeneratePassVectors();

private:
  unsigned int m_NumberOfLevels{ 5 }; // Holds the number of Reconstruction levels
  unsigned int m_Order{ 3 };          // Holds the order of the wavelet filters
  bool         m_PipelineConstructed{
    false
  }; // Filters instantiated by GenerateOutputInformation() should be instantiated only once

  typename InputImageType::SizeType *                   m_Sizes;      // Holds the size of sub-images at each level
  typename InputImageType::IndexType *                  m_Indices;    // Holds the size of sub-images at each level
  typename std::vector<typename AddFilterType::Pointer> m_AddFilters; // Holds a vector of add filters
  typename std::vector<typename ConvolutionFilterType::Pointer>
    m_ConvolutionFilters; // Holds a vector of convolution filters
  typename std::vector<typename UpsampleImageFilterType::Pointer>
    m_UpsampleFilters; // Holds a vector of Upsample filters
  // Holds a vector of PassVectors. A PassVector has Dimension components, each one storing either "High" or "Low"
  typename std::vector<typename ConvolutionFilterType::PassVector> m_PassVectors;
};

} // namespace rtk

// Include CXX
#ifndef rtk_MANUAL_INSTANTIATION
#  include "rtkReconstructImageFilter.hxx"
#endif

#endif
