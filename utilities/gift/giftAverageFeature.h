/*=========================================================================

  Program:  GIFT Average Feature Weight Generator
  Module:   giftAverageFeature.h
  Language: C++
  Date:     2005/11/24
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftAverageFeature_H
#define __giftAverageFeature_H

//ITK includes


//GIFT includes
#include "giftWeightGenerator.h"

 namespace gift {

/**
 * \class AverageFeature
 * \brief Generators weight maps by setting the weight to be 
 *        (sum of feature values)/(number of features). 
 *
 * \ingroup Image Fusion Weight Generators
 */
template <class TImage>
class AverageFeature
    : public WeightGenerator<TImage>
{
public:
    /** Standard class typedefs. */
    typedef AverageFeature                      Self;
    typedef WeightGenerator<TImage>             Superclass;
    typedef itk::SmartPointer<Self>             Pointer;
    typedef itk::SmartPointer<const Self>       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(AverageFeature, WeightGenerator);

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
    AverageFeature();
    ~AverageFeature(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

private:
    AverageFeature(const Self&);    //purposely not implemented
    void operator=(const Self&);    //purposely not implemented
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftAverageFeature.txx"
#endif

#endif
