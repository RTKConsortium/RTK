/*=========================================================================

  Program:  GIFT Median Window Sample Feature Generator
  Module:   giftMedianWindowSampleActivity.h
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
#ifndef __giftMedianWindowSampleActivity_H
#define __giftMedianWindowSampleActivity_H

//ITK includes
#include "itkMedianImageFilter.h"

//GIFT includes
#include "giftFeatureGenerator.h"


namespace gift {

/**
 * \class MedianWindowSampleActivity
 * \brief An image fusion activity feature which computes the median absolute value 
 *        in the given window size.
 *
 * The MedianWindowSampleActivity computes the activity of each level/band
 * by finding the median absolute value within the given window size. 
 * This feature therefore has the same number of output images, levels, and 
 * bands as the input images, levels, and bands.
 *
 * \ingroup Image Fusion Features
 */
template <class TImage>
class MedianWindowSampleActivity
    : public FeatureGenerator<TImage>
{
public:
    /** Standard class typedefs. */
    typedef MedianWindowSampleActivity          Self;
    typedef FeatureGenerator<TImage>            Superclass;
    typedef itk::SmartPointer<Self>             Pointer;
    typedef itk::SmartPointer<const Self>       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(MedianWindowSampleActivity, FeatureGenerator);

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

    /** Typedef for median filter */
    typedef itk::MedianImageFilter<InputImageType, OutputImageType> 
        InternalMedianFilterType;
    typedef typename InternalMedianFilterType::InputSizeType WindowSizeType;

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
    MedianWindowSampleActivity();
    ~MedianWindowSampleActivity(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

private:
    MedianWindowSampleActivity(const Self&);    //purposely not implemented
    void operator=(const Self&);                //purposely not implemented

    WindowSizeType m_WindowSize;
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftMedianWindowSampleActivity.txx"
#endif

#endif
