/*=========================================================================

  Program:  GIFT Absolute Sample Activity Feature Generator
  Module:   giftAbsSampleActivity.h
  Language: C++
  Date:     2005/11/26
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftAbsSampleActivity_H
#define __giftAbsSampleActivity_H

//ITK includes


//GIFT includes
#include "giftFeatureGenerator.h"


namespace gift {

/**
 * \class AbsSampleActivity
 * \brief An image fusion activity feature based on the absolute sample 
 *        value.
 *
 * The AbsSampleActivity assumes the activity of each level/band
 * is simply the absolute sample value. This feature therefore has 
 * the same number of output images, levels, and bands as the input images, 
 * levels, and bands.
 *
 * \ingroup Image Fusion Features
 */
template <class TImage>
class AbsSampleActivity
    : public FeatureGenerator<TImage>
{
public:
    /** Standard class typedefs. */
    typedef AbsSampleActivity                   Self;
    typedef FeatureGenerator<TImage>            Superclass;
    typedef itk::SmartPointer<Self>             Pointer;
    typedef itk::SmartPointer<const Self>       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(AbsSampleActivity, FeatureGenerator);

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
    
    /** Override function. */
    virtual void SetNumberOfInputImages(unsigned int images)
    {
        //Call super
        Superclass::SetNumberOfInputImages(images);

        //Set output images
        this->SetNumberOfOutputImages(images);
    }

    /** Override function. */
    virtual void SetNumberOfInputLevels(unsigned int levels)
    {
        //Call super
        Superclass::SetNumberOfInputLevels(levels);

        //Set output levels
        this->SetNumberOfOutputLevels(levels);
    }

    /** Override function. */
    virtual void SetNumberOfInputBands(unsigned int bands)
    {
        //Call super
        Superclass::SetNumberOfInputBands(bands);

        //Set output bands
        this->SetNumberOfOutputBands(bands);
    }

protected:
    AbsSampleActivity();
    ~AbsSampleActivity(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

private:
    AbsSampleActivity(const Self&);     //purposely not implemented
    void operator=(const Self&);        //purposely not implemented
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftAbsSampleActivity.txx"
#endif

#endif
