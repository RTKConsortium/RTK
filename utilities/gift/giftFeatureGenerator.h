/*=========================================================================

  Program:  GIFT Feature Generator
  Module:   giftFeatureGenerator.h
  Language: C++
  Date:     2005/11/27
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftFeatureGenerator_H
#define __giftFeatureGenerator_H

//ITK includes

//GIFT includes
#include "giftMultilevelMultibandImageFilter.h"

namespace gift {

/**
 * \class FeatureGenerator
 * \brief Inputs all input images (decomposed or normal) and generates a
 *        feature (activity, match, region, etc...) to guide the image
 *        fusion process..
 *
 * This is an abstract class. Subclasses must override the GenerateData()
 * method.
 *
 * \ingroup Image Fusion Weight Generator
 */
template <class TImage>
class FeatureGenerator
    : public MultilevelMultibandImageFilter<TImage>
{
public:
    /** Standard class typedefs. */
    typedef FeatureGenerator                        Self;
    typedef MultilevelMultibandImageFilter<TImage>  Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Run-time type information (and related methods). */
    itkTypeMacro(FeatureGenerator, MultilevelMultibandImageFilter);

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
    
protected:
    FeatureGenerator();
    ~FeatureGenerator(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData() = 0;

private:
    FeatureGenerator(const Self&);      //purposely not implemented
    void operator=(const Self&);        //purposely not implemented
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftFeatureGenerator.txx"
#endif

#endif
