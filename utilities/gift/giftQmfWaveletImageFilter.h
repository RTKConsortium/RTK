/*=========================================================================

  Program:  GIFT QMF Wavelet Image Filter
  Module:   giftQmfWaveletImageFilter.h
  Language: C++
  Date:     2005/11/16
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftQmfWaveletImageFilter_H
#define __giftQmfWaveletImageFilter_H

//ITK includes
#include "itkMacro.h"
#include "itkProgressReporter.h"

//GIFT includes
#include "giftBiorthogonalWaveletOperator.h"
#include "giftMultilevelMultibandImageFilter.h"


namespace gift {

/**
 * \class QmfWaveletImageFilter
 * \brief An image filter that deconstructs OR reconstructs an 
 *        image based on quadrature mirror filter (QMF) wavelets.
 *
 * This object can either have a single image as input (in reconstruction mode)
 * or multiple images (that is level/band images) as input (in deconstruction 
 * mode).
 *
 * The number of images is 1, and number of bands (per level) is 4.
 * The user can set the number of levels. 
 *
 * \ingroup Wavelet Image Filters
 */
template <class TImage, class TWavelet>
class QmfWaveletImageFilter
    : public MultilevelMultibandImageFilter<TImage>
{
public:
    /** Standard class typedefs. */
    typedef QmfWaveletImageFilter                   Self;
    typedef MultilevelMultibandImageFilter<TImage>  Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(QmfWaveletImageFilter, MultilevelMultibandImageFilter);

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
    
    /** Get the number of multi-level/multi-band input images. */
    virtual unsigned int GetNumberOfInputImages()
    {
        return 1;
    }

    /** Get the number of multi-level/multi-band output images. */
    virtual unsigned int GetNumberOfOutputImages()
    {
        return 1;
    }

    /** Set the number of levels.
     * (NOTE: that the number of input and output levels are the same.
     */
    void SetNumberOfLevels(unsigned int levels)
    {
        this->SetNumberOfInputLevels(levels);
        this->SetNumberOfOutputLevels(levels);
    }

    /** Gets the number of levels.
     * (NOTE: that the number of input and output levels are the same.
     */
    unsigned int GetNumberOfLevels()
    {
        return this->GetNumberOfInputLevels();
    }

    /** Override method to set the number of input/output bands */ 
    virtual void SetDeconstruction()
    {
        //Call super
        Superclass::SetDeconstruction();
        
        //Change the number of bands
        //NOTE: these methods will call ModifyInputOutputStorage()
        this->SetNumberOfInputBands(1);
        this->SetNumberOfOutputBands((unsigned int)pow(2.0, (int)ImageDimension));
    }

    /** Override method to set the number of input/output bands */ 
    virtual void SetReconstruction()
    {
        //Call super
        Superclass::SetReconstruction();
        
        //Change the number of bands
        //NOTE: these methods will call ModifyInputOutputStorage()
        this->SetNumberOfInputBands((unsigned int)pow(2.0, (int)ImageDimension));
        this->SetNumberOfOutputBands(1);
    }

    /** We require a larger input requested region than the output 
     *  requested regions to accomdate the filtering operations. 
     *  As such, we reimplement GenerateInputRequestedRegion(). */
    virtual void GenerateInputRequestedRegion();

    /** We produce images which may be of different resolution and different 
     *  pixel spacing than the input image.
     *  As such, we reimplement GenerateOutputInformation() in order to 
     *  inform the pipeline execution model.  */
    virtual void GenerateOutputInformation();

    /** FeatureGenerators produce images which are of different size.
     *  As such, we reimplement GenerateOutputInformation() in order to 
     *  inform the pipeline execution model.
     *  This implementation uses the requested region foreach of the outputs.
     */
    virtual void GenerateOutputRequestedRegion(itk::DataObject *output);

    /** Sets the wavelet used to deconstruct/reconstruct the image pyramid */
    void SetWavelet(TWavelet wavelet)
    {
        this->m_Wavelet = wavelet;
    }

protected:
    QmfWaveletImageFilter();
    ~QmfWaveletImageFilter(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

    /** Adds the high-pass, low-pass filters to the inputs for a given dimension */ 
    void AddFiltersForDimension(unsigned int idim, std::vector<itk::DataObject::Pointer>& inputs, itk::ProgressReporter &progress);

private:
    QmfWaveletImageFilter(const Self&);     //purposely not implemented
    void operator=(const Self&);            //purposely not implemented

    TWavelet m_Wavelet;
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftQmfWaveletImageFilter.txx"
#endif

#endif
