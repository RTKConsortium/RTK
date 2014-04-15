/*=========================================================================

  Program:  GIFT Weight Combiner
  Module:   giftWeightCombiner.h
  Language: C++
  Date:     2005/11/23
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftWeightCombiner_H
#define __giftWeightCombiner_H

//ITK includes


//GIFT includes
#include "giftMultilevelMultibandImageFilter.h"

 namespace gift {

/**
 * \class WeightCombiner
 * \brief Responsible for combining the weight maps and deconstructed inputs 
 *        to form a multi-level/multi-band image with NumberOfOutputImages = 1; 
 *
 * \ingroup Image Fusion Combiners
 */
template <class TImage>
class WeightCombiner
    : public MultilevelMultibandImageFilter<TImage>
{
public:
    /** Standard class typedefs. */
    typedef WeightCombiner                          Self;
    typedef MultilevelMultibandImageFilter<TImage>  Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Run-time type information (and related methods). */
    itkTypeMacro(WeightCombiner, MultilevelMultibandImageFilter);

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
    
    /** Sets the number of input and output images.
     *  images = the number of images to fuse. */
    virtual void SetNumberOfImages(unsigned int images)
    {
        //NOTE: *2 is because the WeightCombiner must receive an input image
        //      AND a weight map for each image to fuse.
        Superclass::SetNumberOfInputImages(images*2);
        Superclass::SetNumberOfOutputImages(1);
    }

    /** Sets the number of input and output levels. */
    virtual void SetNumberOfLevels(unsigned int levels)
    {
        Superclass::SetNumberOfInputLevels(levels);
        Superclass::SetNumberOfOutputLevels(levels);
    }

    /** Set the number of input and output bands. */
    virtual void SetNumberOfBands(unsigned int bands)
    {
        Superclass::SetNumberOfInputBands(bands);
        Superclass::SetNumberOfOutputBands(bands);
    }

protected:
    WeightCombiner();
    ~WeightCombiner(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData() = 0;

private:
    WeightCombiner(const Self&);        //purposely not implemented
    void operator=(const Self&);        //purposely not implemented
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftWeightCombiner.txx"
#endif

#endif
