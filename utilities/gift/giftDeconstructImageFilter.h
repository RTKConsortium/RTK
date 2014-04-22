/*=========================================================================

  Program:  GIFT Multi-level Multi-band Image Filter
  Module:   giftDeconstructImageFilter.h
  Language: C++
  Date:     2005/11/22
  Version:  0.2
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftDeconstructImageFilter_H
#define __giftDeconstructImageFilter_H

//Includes
#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkPadImageFilter.h>
#include <itkFFTConvolutionImageFilter.h>

#include "giftDownsampleImageFilter.h"

namespace gift {

/**
 * \class DeconstructImageFilter
 * \brief An image filter that deconstructs an image.
 *
 * NOTE: Level=0 is the TOP level (original image).
 *
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
    typedef itk::PadImageFilter<InputImageType>             PadFilterType;
    typedef itk::FFTConvolutionImageFilter<InputImageType>  ConvolutionFilterType;
    typedef gift::DownsampleImageFilter<InputImageType>     DownsampleImageFilter;

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

    /** Gets the output image at the given image/level/band */
    OutputImageType* GetOutputByLevelBand(unsigned int level, unsigned int band);
    
    /** Gets the output image (as an itk::DataObject*) at the given image/level/band */
    itk::DataObject* GetDataObjectOutputByLevelBand(unsigned int level, unsigned int band);

    /** Sets the output image at the given level/band */
    void SetOutputByLevelBand(unsigned int level, unsigned int band, const OutputImageType * input);
    
    /** Gets the output image at the given level/band */
    const OutputImageType* GetOutputByLevelBand(unsigned int level, unsigned int band);

    /** Gets the number of bands for a given level */
    unsigned int GetNumberOfOutputBands(unsigned int level);

    /** DeconstructImageFilters produce images which are of different size
     *  than the first input image (copying the information from the first input 
     *  to all outputs is the default implementation of GenerateOutputInformation() 
     *  in itkProcessObject.txx).
     *  As such, we reimplement GenerateOutputInformation() in order to 
     *  inform the pipeline execution model.  
     */
    virtual void GenerateOutputInformation();

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

    /** Computes the n-D kernels from the 1-D ones */
    virtual typename InputImageType::Pointer ComputeNdKernel(unsigned int index);
    
    /** Calculates the number of ProcessObject output images */
    virtual unsigned int CalculateNumberOfOutputs();

    /** Converts an output index to image/level/band */
    void ConvertOutputIndexToLevelBand(unsigned int index, unsigned int &level, unsigned int &band);

    /** Converts an output image/level/band to an index */
    unsigned int ConvertOutputLevelBandToIndex(unsigned int level, unsigned int band);

private:
    DeconstructImageFilter(const Self&);    //purposely not implemented
    void operator=(const Self&);                    //purposely not implemented

    unsigned int m_NumberOfLevels;     //Holds the number of deconstruction levels
    typename PadFilterType::Pointer           m_PadFilter;
    typename ConvolutionFilterType::Pointer   *m_ConvolutionFilters; //Holds an array convolution filters
    typename ConvolutionFilterType::Pointer   *m_DownsampleFilters; //Holds an array of downsample filters
    typename TImage::Pointer                  *m_Kernels; //Holds the kernels
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftDeconstructImageFilter.txx"
#endif

#endif
