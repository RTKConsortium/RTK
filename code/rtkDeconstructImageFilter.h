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

#ifndef rtkDeconstructImageFilter_h
#define rtkDeconstructImageFilter_h

//Includes
#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkMirrorPadImageFilter.h>

#include "rtkDaubechiesWaveletsConvolutionImageFilter.h"
#include "rtkDownsampleImageFilter.h"

namespace rtk {

/**
 * \class DeconstructImageFilter
 * \brief An image filter that deconstructs an image using
 * Daubechies wavelets.
 *
 * This filter is inspired from Dan Mueller's GIFT package
 * http://www.insight-journal.org/browse/publication/103
 *
 *
 * \dot
 * digraph DeconstructImageFilter {
 *
 * Input [ label="Input (here, 2D)"];
 * Input [shape=Mdiamond];
 * Output0 [label="Output 0"];
 * Output0 [shape=Mdiamond];
 * Output1 [label="Output 1"];
 * Output1 [shape=Mdiamond];
 * Output2 [label="Output 2"];
 * Output2 [shape=Mdiamond];
 * Output3 [label="Output 3"];
 * Output3 [shape=Mdiamond];
 * Output4 [label="Output 4"];
 * Output4 [shape=Mdiamond];
 * Output5 [label="Output 5"];
 * Output5 [shape=Mdiamond];
 * Output6 [label="Output 6"];
 * Output6 [shape=Mdiamond];
 *
 * node [shape=box];
 * Pad0 [ label="itk::MirrorPadImageFilter" URL="\ref itk::MirrorPadImageFilter"];
 * Pad1 [ label="itk::MirrorPadImageFilter" URL="\ref itk::MirrorPadImageFilter"];
 * Conv0 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Lowpass, Lowpass)" URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv1 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Lowpass, Highpass)" URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv2 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Highpass, Lowpass)" URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv3 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Highpass, Highpass)" URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv4 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Lowpass, Lowpass)" URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv5 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Lowpass, Highpass)" URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv6 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Highpass, Lowpass)" URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Conv7 [ label="rtk::DaubechiesWaveletsConvolutionImageFilter (Highpass, Highpass)" URL="\ref rtk::DaubechiesWaveletsConvolutionImageFilter"];
 * Down0 [ label="rtk::DownsampleImageFilter (by 2)" URL="\ref rtk::DownsampleImageFilter"];
 * Down1 [ label="rtk::DownsampleImageFilter (by 2)" URL="\ref rtk::DownsampleImageFilter"];
 * Down2 [ label="rtk::DownsampleImageFilter (by 2)" URL="\ref rtk::DownsampleImageFilter"];
 * Down3 [ label="rtk::DownsampleImageFilter (by 2)" URL="\ref rtk::DownsampleImageFilter"];
 * Down4 [ label="rtk::DownsampleImageFilter (by 2)" URL="\ref rtk::DownsampleImageFilter"];
 * Down5 [ label="rtk::DownsampleImageFilter (by 2)" URL="\ref rtk::DownsampleImageFilter"];
 * Down6 [ label="rtk::DownsampleImageFilter (by 2)" URL="\ref rtk::DownsampleImageFilter"];
 * Down7 [ label="rtk::DownsampleImageFilter (by 2)" URL="\ref rtk::DownsampleImageFilter"];
 * AfterPad0 [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterPad1 [label="", fixedsize="false", width=0, height=0, shape=none];
 * Input -> Pad1;
 * Pad1 -> AfterPad1 [arrowhead=none];
 * AfterPad1 -> Conv4;
 * AfterPad1 -> Conv5;
 * AfterPad1 -> Conv6;
 * AfterPad1 -> Conv7;
 * Conv4 -> Down4;
 * Conv5 -> Down5;
 * Conv6 -> Down6;
 * Conv7 -> Down7;
 * Down4 -> Pad0;
 * Pad0 -> AfterPad0 [arrowhead=none];
 * AfterPad0 -> Conv0;
 * AfterPad0 -> Conv1;
 * AfterPad0 -> Conv2;
 * AfterPad0 -> Conv3;
 * Conv0 -> Down0;
 * Conv1 -> Down1;
 * Conv2 -> Down2;
 * Conv3 -> Down3;
 * Down0 -> Output0;
 * Down1 -> Output1;
 * Down2 -> Output2;
 * Down3 -> Output3;
 * Down5 -> Output4;
 * Down6 -> Output5;
 * Down7 -> Output6;
 * }
 * \enddot
 *
 * \test rtkwaveletstest.cxx
 *
 * \author Cyril Mory
 */
template <class TImage>
class DeconstructImageFilter
    : public itk::ImageToImageFilter<TImage, TImage>
{
public:
    /** Standard class typedefs. */
    typedef DeconstructImageFilter                  Self;
    typedef itk::ImageToImageFilter<TImage,TImage>  Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(DeconstructImageFilter, ImageToImageFilter)

    /** ImageDimension enumeration. */
    itkStaticConstMacro(ImageDimension, unsigned int, TImage::ImageDimension);

    /** Inherit types from Superclass. */
    typedef typename Superclass::InputImageType         InputImageType;
    typedef typename Superclass::OutputImageType        OutputImageType;
    typedef typename Superclass::InputImagePointer      InputImagePointer;
    typedef typename Superclass::OutputImagePointer     OutputImagePointer;
    typedef typename Superclass::InputImageConstPointer InputImageConstPointer;
    typedef typename TImage::PixelType                  PixelType;
    typedef typename TImage::InternalPixelType          InternalPixelType;

    /** Typedefs for pipeline's subfilters */
    typedef itk::MirrorPadImageFilter<InputImageType, InputImageType>             PadFilterType;
    typedef rtk::DaubechiesWaveletsConvolutionImageFilter<InputImageType>        ConvolutionFilterType;
    typedef rtk::DownsampleImageFilter<InputImageType>           DownsampleImageFilterType;

    /** Set the number of input levels. */
    virtual void SetNumberOfLevels(unsigned int levels)
    {
      this->m_NumberOfLevels = levels;
      this->ModifyInputOutputStorage();
    }

    /** Get the number of input levels (per image). */
    virtual unsigned int GetNumberOfLevels()
    {
      return this->m_NumberOfLevels;
    }

    /** DeconstructImageFilter produces images which are of different size
     *  than the input image. As such, we reimplement GenerateOutputInformation()
     *  in order to inform the pipeline execution model.
     */
    void GenerateOutputInformation() ITK_OVERRIDE;

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Get/Set the order of the wavelet filters */
    itkGetMacro(Order, unsigned int)
    itkSetMacro(Order, unsigned int)

    /** Get the size of each convolution filter's output
     * This is required because the downsampling implies
     * a loss of information on the size (both 2n+1 and 2n
     * are downsampled to n), and the upsampling filters
     * used in the reconstruction process need this
     * information.
     */
    typename InputImageType::SizeType* GetSizes()
    {
    return m_Sizes.data();
    }

    /** Get the index of each convolution filter's output
     * This is required because the downsampling implies
     * a loss of information on the index (both 2n+1 and 2n
     * are downsampled to n), and the upsampling filters
     * used in the reconstruction process need this
     * information.
     */
    typename InputImageType::IndexType* GetIndices()
    {
    return m_Indices.data();
    }

protected:
    DeconstructImageFilter();
    ~DeconstructImageFilter() {}
    void PrintSelf(std::ostream&os, itk::Indent indent) const ITK_OVERRIDE;

    /** Modifies the storage for Input and Output images.
      * Should be called after changes to levels, bands,
      * deconstruct, reconstruct, etc... */
    void ModifyInputOutputStorage();

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

    /** Calculates the number of ProcessObject output images */
    virtual unsigned int CalculateNumberOfOutputs();

    /** Creates and sets the kernel sources to generate all kernels. */
    void GeneratePassVectors();

private:
    DeconstructImageFilter(const Self&);    //purposely not implemented
    void operator=(const Self&);                    //purposely not implemented

    unsigned int m_NumberOfLevels;        // Holds the number of deconstruction levels
    unsigned int m_Order;                 // Holds the order of the wavelet filters
    bool         m_PipelineConstructed;   // Filters instantiated by GenerateOutputInformation() should be instantiated only once

    typename std::vector<typename InputImageType::SizeType>             m_Sizes; //Holds the size of sub-images at each level
    typename std::vector<typename InputImageType::IndexType>            m_Indices; //Holds the size of sub-images at each level
    typename std::vector<typename PadFilterType::Pointer>               m_PadFilters; //Holds a vector of padding filters
    typename std::vector<typename ConvolutionFilterType::Pointer>       m_ConvolutionFilters; //Holds a vector of convolution filters
    typename std::vector<typename DownsampleImageFilterType::Pointer>   m_DownsampleFilters; //Holds a vector of downsample filters
    //Holds a vector of PassVectors. A PassVector has Dimension components, each one storing either "High" or "Low"
    typename std::vector<typename ConvolutionFilterType::PassVector>    m_PassVectors;
};

}// namespace rtk

//Include CXX
#ifndef rtk_MANUAL_INSTANTIATION
#include "rtkDeconstructImageFilter.hxx"
#endif

#endif
