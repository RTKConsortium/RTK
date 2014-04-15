/*=========================================================================

  Program:  GIFT Maximum Window Sample Feature Generator
  Module:   giftMaximumWindowSampleActivity.h
  Language: C++
  Date:     2006/06/03
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2006 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftMaximumWindowSampleActivity_H
#define __giftMaximumWindowSampleActivity_H

//ITK includes


//GIFT includes
#include "giftFeatureGenerator.h"


namespace gift {

/**
 * \class MaximumWindowSampleActivity
 * \brief An image fusion activity feature which computes the maximum absolute value 
 *        in the given window size.
 *
 * The MaximumWindowSampleActivity computes the activity of each level/band
 * by finding the maximum absolute value within the given window size. 
 * This feature therefore has the same number of output images, levels, and 
 * bands as the input images, levels, and bands.
 *
 * \ingroup Image Fusion Features
 */
template <class TImage>
class MaximumWindowSampleActivity
    : public FeatureGenerator<TImage>
{
public:
    /** Standard class typedefs. */
    typedef MaximumWindowSampleActivity         Self;
    typedef FeatureGenerator<TImage>            Superclass;
    typedef itk::SmartPointer<Self>             Pointer;
    typedef itk::SmartPointer<const Self>       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(MaximumWindowSampleActivity, FeatureGenerator);

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

    /** Typedef for window filter */
    typedef typename InputImageType::SizeType           WindowSizeType;

    /** Set the size of the window */
    itkSetMacro(WindowSize, WindowSizeType);

    /** Set the radius of the window 
     *  NOTE:   0 = [0, 0] = 1x1 window = the same as AbsSampleActivity
     *          1 = [1, 1] = 3x3 window
     *          2 = [2, 2] = 5x5 window
     */
    void SetWindowRadius(unsigned int windowRadius)
    {
        WindowSizeType windowSize;
        windowSize.Fill(windowRadius);
        this->SetWindowSize(windowSize);
    }

    /** Get the window size */
    itkGetConstReferenceMacro(WindowSize, WindowSizeType);
    
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
    MaximumWindowSampleActivity();
    ~MaximumWindowSampleActivity(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

private:
    MaximumWindowSampleActivity(const Self&);   //purposely not implemented
    void operator=(const Self&);                //purposely not implemented

    WindowSizeType m_WindowSize;
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftMaximumWindowSampleActivity.txx"
#endif

#endif
