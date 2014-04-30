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

#ifndef __rtkDeconstructImageFilter_H
#define __rtkDeconstructImageFilter_H

//Includes
#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkMirrorPadImageFilter.h>
#include <itkFFTConvolutionImageFilter.h>

#include "rtkDaubechiesWaveletsKernelSource.h"
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
 * \author Cyril Mory
 */
template <class TImage>
class DeconstructImageFilter
    : public itk::ImageToImageFilter<TImage, TImage>
{
public: 
    /** Standard class typedefs. */
    typedef DeconstructImageFilter          Self;
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
    typedef itk::FFTConvolutionImageFilter<InputImageType>        ConvolutionFilterType;
    typedef rtk::DownsampleImageFilter<InputImageType>           DownsampleImageFilterType;
    typedef rtk::DaubechiesWaveletsKernelSource<InputImageType>  KernelSourceType;

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
    virtual void GenerateOutputInformation();

    virtual void GenerateInputRequestedRegion();

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
    ~DeconstructImageFilter() {};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Modifies the storage for Input and Output images.
      * Should be called after changes to levels, bands, 
      * deconstruct, reconstruct, etc... */
    void ModifyInputOutputStorage();

    /** Does the real work. */
    virtual void GenerateData();

    /** Calculates the number of ProcessObject output images */
    virtual unsigned int CalculateNumberOfOutputs();

    /** Creates and sets the kernel sources to generate all kernels. */
    void GenerateAllKernelSources();

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
    typename std::vector<typename KernelSourceType::Pointer>            m_KernelSources; //Holds a vector of kernel sources
};

}// namespace rtk

//Include CXX
#ifndef rtk_MANUAL_INSTANTIATION
#include "rtkDeconstructImageFilter.txx"
#endif

#endif
