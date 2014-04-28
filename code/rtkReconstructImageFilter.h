/*=========================================================================

  Program:  rtk Multi-level Multi-band Image Filter
  Module:   rtkReconstructImageFilter.h
  Language: C++
  Date:     2005/11/22
  Version:  0.2
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See rtkCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __rtkReconstructImageFilter_H
#define __rtkReconstructImageFilter_H

//Includes
#include <itkImageToImageFilter.h>
#include <itkMacro.h>
#include <itkNaryAddImageFilter.h>
#include <itkFFTConvolutionImageFilter.h>

#include "rtkDaubechiesWaveletsKernelSource.h"
#include "rtkUpsampleImageFilter.h"

// For debugging
#include <itkImageFileWriter.h>
#include <string>
#include <sstream>

namespace rtk {

/**
 * \class ReconstructImageFilter
 * \brief An image filter that reconstructs an image using
 * Daubechies wavelets.
 *
 * \author Cyril Mory
 */
template <class TImage>
class ReconstructImageFilter
    : public itk::ImageToImageFilter<TImage, TImage>
{
public: 
    /** Standard class typedefs. */
    typedef ReconstructImageFilter          Self;
    typedef itk::ImageToImageFilter<TImage,TImage>  Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)
    
    /** Run-time type information (and related methods). */
    itkTypeMacro(ReconstructImageFilter, ImageToImageFilter)

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
    typedef itk::NaryAddImageFilter<InputImageType, InputImageType>   AddFilterType;
    typedef itk::FFTConvolutionImageFilter<InputImageType>        ConvolutionFilterType;
    typedef rtk::UpsampleImageFilter<InputImageType>             UpsampleImageFilterType;
    typedef rtk::DaubechiesWaveletsKernelSource<InputImageType>  KernelSourceType;

    typedef itk::ImageFileWriter<InputImageType>          WriterType;

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

    /** ReconstructImageFilter produces images which are of different size
     *  than the input image. As such, we reimplement GenerateOutputInformation()
     *  in order to inform the pipeline execution model.
     */
    virtual void GenerateOutputInformation();


    /** ReconstructImageFilter requests the largest possible region of all its inputs.
     */
    virtual void GenerateInputRequestedRegion();

    /** ReconstructImageFilter uses input images of different sizes, therefore the
     * VerifyInputInformation method has to be reimplemented.
     */
    virtual void VerifyInputInformation() {}

    void SetSizes(typename InputImageType::SizeType *sizesVector)
    {
    m_Sizes = sizesVector;
    }

    /** Get/Set the order of the wavelet filters */
    itkGetMacro(Order, unsigned int)
    itkSetMacro(Order, unsigned int)

protected:
    ReconstructImageFilter();
    ~ReconstructImageFilter() {}

    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Modifies the storage for Input and Output images.
      * Should be called after changes to levels, bands, 
      * Reconstruct, reconstruct, etc... */
    void ModifyInputOutputStorage();

    /** Does the real work. */
    virtual void GenerateData();

    /** Calculates the number of ProcessObject output images */
    virtual unsigned int CalculateNumberOfInputs();

    /** Creates and sets the kernel sources to generate all kernels. */
    void GenerateAllKernelSources();

private:
    ReconstructImageFilter(const Self&);    //purposely not implemented
    void operator=(const Self&);                    //purposely not implemented

    unsigned int m_NumberOfLevels;     //Holds the number of Reconstruction levels
    unsigned int m_Order;             // Holds the order of the wavelet filters
    typename InputImageType::SizeType             *m_Sizes; //Holds the size of sub-images at each level

    typename std::vector<typename AddFilterType::Pointer>               m_AddFilters; //Holds a vector of add filters
    typename std::vector<typename ConvolutionFilterType::Pointer>       m_ConvolutionFilters; //Holds a vector of convolution filters
    typename std::vector<typename UpsampleImageFilterType::Pointer>     m_UpsampleFilters; //Holds a vector of Upsample filters
    typename std::vector<typename KernelSourceType::Pointer>            m_KernelSources; //Holds a vector of kernel sources

    typename WriterType::Pointer                                        m_Writer; //For debugging
};

}// namespace rtk

//Include CXX
#ifndef rtk_MANUAL_INSTANTIATION
#include "rtkReconstructImageFilter.txx"
#endif

#endif
