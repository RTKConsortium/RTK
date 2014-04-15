/*=========================================================================

  Program:  GIFT Sample Activity 
  Module:   giftSampleActivity.h
  Language: C++
  Date:     2005/11/22
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftSampleActivity_H
#define __giftSampleActivity_H

//ITK includes


//GIFT includes
#include "giftFeatureGenerator.h"


namespace gift {

/**
 * \class SampleActivity
 * \brief An image fusion activity feature based on the sample value.
 *
 * The SampleActivity assumes the activity of each level/band
 * is simply the sample value. Therefore it simply copies the input 
 * coefficients as the activity feature measure. This feature therefore has 
 * the same number of output images, levels, and bands as the input images, 
 * levels, and bands.
 *
 * \ingroup Image Fusion Features
 */
template <class TImage>
class SampleActivity
    : public FeatureGenerator<TImage>
{
public:
    /** Standard class typedefs. */
    typedef SampleActivity                  Self;
    typedef FeatureGenerator<TImage>        Superclass;
    typedef itk::SmartPointer<Self>         Pointer;
    typedef itk::SmartPointer<const Self>   ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(SampleActivity, FeatureGenerator);

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
    SampleActivity();
    ~SampleActivity(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

private:
    SampleActivity(const Self&);    //purposely not implemented
    void operator=(const Self&);    //purposely not implemented
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftSampleActivity.txx"
#endif

#endif
