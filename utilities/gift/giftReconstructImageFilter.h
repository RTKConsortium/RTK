/*=========================================================================

  Program:  GIFT Multi-level Multi-band Image Filter
  Module:   giftMultilevelMultibandImageFilter.h
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
#ifndef __giftMultilevelMultibandImageFilter_H
#define __giftMultilevelMultibandImageFilter_H

//Includes
#include "itkImageToImageFilter.h"
#include "itkMacro.h"

namespace gift {

/**
 * \class MultilevelMultibandImageFilter
 * \brief An image filter that deconstructs OR reconstructs a 
 *        multi-level multi-band image.
 *
 * This class can either have multiple or single input(s) and
 * produce multiple or single output(s).
 *
 * NOTE: Level=0 is the TOP level (original image).
 *
 * \ingroup Multi-level multi-band Image Filters
 */
template <class TImage>
class MultilevelMultibandImageFilter
    : public itk::ImageToImageFilter<TImage, TImage>
{
public: 
    /** Standard class typedefs. */
    typedef MultilevelMultibandImageFilter          Self;
    typedef itk::ImageToImageFilter<TImage,TImage>  Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Method for creation through the object factory. */
    //itkNewMacro(Self);
    
    /** Run-time type information (and related methods). */
    itkTypeMacro(MultilevelMultibandImageFilter, ImageToImageFilter);

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
    
    /** Set the number of input images. */
    virtual void SetNumberOfInputImages(unsigned int images)
    {
        this->m_NumberOfInputImages = images;
        this->ModifyInputOutputStorage();
    }

    /** Set the number of output images. */
    virtual void SetNumberOfOutputImages(unsigned int images)
    {
        this->m_NumberOfOutputImages = images;
        this->ModifyInputOutputStorage();
    }

    /** Get the number of input images. */
    virtual unsigned int GetNumberOfInputImages()
    {
        return this->m_NumberOfInputImages;
    }

    /** Get the number of output images. */
    virtual unsigned int GetNumberOfOutputImages()
    {
        return this->m_NumberOfOutputImages;
    }

    /** Set the number of input levels. */
    virtual void SetNumberOfInputLevels(unsigned int levels)
    {
        this->m_NumberOfInputLevels = levels;
        this->ModifyInputOutputStorage();
    }

    /** Set the number of output levels. */
    virtual void SetNumberOfOutputLevels(unsigned int levels)
    {
        this->m_NumberOfOutputLevels = levels;
        this->ModifyInputOutputStorage();
    }

    /** Get the number of input levels (per image). */
    virtual unsigned int GetNumberOfInputLevels()
    {
        return this->m_NumberOfInputLevels;
    }

    /** Get the number of output levels (per image). */
    virtual unsigned int GetNumberOfOutputLevels()
    {
        return this->m_NumberOfOutputLevels;
    }

    /** Set the number of input bands. */
    virtual void SetNumberOfInputBands(unsigned int bands)
    {
        this->m_NumberOfInputBands = bands;
        this->ModifyInputOutputStorage();
    }

    /** Set the number of output bands. */
    virtual void SetNumberOfOutputBands(unsigned int bands)
    {
        this->m_NumberOfOutputBands = bands;
        this->ModifyInputOutputStorage();
    }

    /** Get the number of input bands (per level). */
    virtual unsigned int GetNumberOfInputBands()
    {
        return this->m_NumberOfInputBands;
    }

    /** Get the number of output bands (per level). */
    virtual unsigned int GetNumberOfOutputBands()
    {
        return this->m_NumberOfOutputBands;
    }

    /** 
    * Enumerate the multi-level/multi-band image types
    */
    enum Type 
    {
        Deconstruction, //One input, multi-level/multi-band output
        Reconstruction, //Multi-level/multi-band input, one output
        Merge,          //Merges two (or more) multi-level/multi-band images into one
        User            //A user defined structure
    };

    /** Get the type of this filter */
    Type GetType()
    {
        return this->m_Type;
    }

    /** Sets that this filter deconstructs an image into a pyramid.
     *  This means that only 1 image can be handled at a time.
     */
    virtual void SetDeconstruction()
    {
        this->m_Type = Self::Deconstruction;
        this->m_NumberOfInputImages = 1;
        this->m_NumberOfOutputImages = 1;
        this->ModifyInputOutputStorage();
    }

    /** Sets that this filter reconstructs an image from a pyramid.
     *  This means that only 1 image can be handled at a time.
     */
    virtual void SetReconstruction()
    {
        this->m_Type = Self::Reconstruction;
        this->m_NumberOfInputImages = 1;
        this->m_NumberOfOutputImages = 1;
        this->ModifyInputOutputStorage();
    }

    /** Sets that this filter merges two (or more)
     *  multi-level/multi-band image into 1 multi-level/multi-band image */
    virtual void SetMerge(unsigned int numberOfInputImages)
    {
        this->m_Type = Self::Merge;
        this->m_NumberOfInputImages = numberOfInputImages;
        this->m_NumberOfOutputImages = 1;
        this->ModifyInputOutputStorage();
    }

    /** Sets this filter as a user defined type */
    virtual void SetUser(unsigned int numberOfInputImages, 
                         unsigned int numberOfOutputImages)
    {
        this->m_Type = Self::User;
        this->m_NumberOfInputImages = numberOfInputImages;
        this->m_NumberOfOutputImages = numberOfOutputImages;
        this->ModifyInputOutputStorage();
    }

    /** Returns if this filter is responsible for deconstructing 
        an image into a pyramid */
    bool IsDeconstruction()
    {
        return (this->m_Type == Self::Deconstruction);  
    }

    /** Returns if this filter is responsible for reconstructing 
        an image from a pyramid */
    bool IsReconstruction()
    {
        return (this->m_Type == Self::Reconstruction);  
    }

    /** Returns if this filter takes many images and returns many images */
    bool IsMergeType()
    {
        return (this->m_Type == Self::Merge);
    }

    /** Returns if this filter is arbitrarily defined by the user */
    bool IsUserType()
    {
        return (this->m_Type == Self::User);
    }

    /** Gets the output image at the given image/level/band */
    OutputImageType* GetOutputByImageLevelBand(unsigned int image, unsigned int level, unsigned int band);
    
    /** Gets the output image (as an itk::DataObject*) at the given image/level/band */
    itk::DataObject* GetDataObjectOutputByImageLevelBand(unsigned int image, unsigned int level, unsigned int band);

    /** Sets the input image at the given image/level/band */
    void SetInputByImageLevelBand(unsigned int image, unsigned int level, unsigned int band, const InputImageType * input);
    
    /** Getsets the input image at the given image/level/band */
    const InputImageType* GetInputByImageLevelBand(unsigned int image, unsigned int level, unsigned int band);

    /** The inputs are different sizes so we need to reimplement. 
     *  GenerateInputRequestedRegion() to set the requested region. */
    virtual void GenerateInputRequestedRegion();

    /** MultilevelMultibandImageFilters produce images which are of different size 
     *  than the first input image (copying the information from the first input 
     *  to all outputs is the default implementation of GenerateOutputInformation() 
     *  in itkProcessObject.txx).
     *  As such, we reimplement GenerateOutputInformation() in order to 
     *  inform the pipeline execution model.  
     *  This implementation assumes that
     *  NumberOfInputImages/Levels/Bands = NumberOfOutputImages/Levels/Bands. If
     *  this is not the case, specific subclasses must override this method.
     */
    virtual void GenerateOutputInformation();

    /** MultilevelMultibandImageFilters produce images which are of different size.
     *  As such, we reimplement GenerateOutputInformation() in order to 
     *  inform the pipeline execution model.
     *  This implementation uses the requested region foreach of the outputs.
     */
    virtual void GenerateOutputRequestedRegion(itk::DataObject *output);


protected:
    MultilevelMultibandImageFilter();
    ~MultilevelMultibandImageFilter() {};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Modifies the storage for Input and Output images.
      * Should be called after changes to levels, bands, 
      * deconstruct, reconstruct, etc... */
    void ModifyInputOutputStorage();

    /** Calculates the number of ProcessObject input images */
    virtual unsigned int CalculateNumberOfInputs();
    
    /** Calculates the number of ProcessObject output images */
    virtual unsigned int CalculateNumberOfOutputs();

    /** Converts an input index to image/level/band */
    void ConvertInputIndexToImageLevelBand(unsigned int index, unsigned int &image, unsigned int &level, unsigned int &band);
    
    /** Converts an output index to image/level/band */
    void ConvertOutputIndexToImageLevelBand(unsigned int index, unsigned int &image, unsigned int &level, unsigned int &band);

    /** Converts an input image/level/band to an index */
    unsigned int ConvertInputImageLevelBandToIndex(unsigned int image, unsigned int level, unsigned int band);

    /** Converts an output image/level/band to an index */
    unsigned int ConvertOutputImageLevelBandToIndex(unsigned int image, unsigned int level, unsigned int band);

private:
    MultilevelMultibandImageFilter(const Self&);    //purposely not implemented
    void operator=(const Self&);                    //purposely not implemented

    unsigned int m_NumberOfInputImages;     //Holds the total number of multi-level/multi-band input images
    unsigned int m_NumberOfInputLevels;     //Holds the number of levels in each input image
    unsigned int m_NumberOfInputBands;      //Holds the number of bands in each level for each input image.
    
    unsigned int m_NumberOfOutputImages;    //Holds the total number of multi-level/multi-band output images
    unsigned int m_NumberOfOutputLevels;    //Holds the number of levels in each output image
    unsigned int m_NumberOfOutputBands;     //Holds the number of bands in each level for each output image.
    
    Type m_Type;                            //Holds type of the filter.
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftMultilevelMultibandImageFilter.txx"
#endif

#endif
