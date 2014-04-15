/*=========================================================================

  Program:  GIFT Select Feature Weight Generator
  Module:   giftSelectFeature.h
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
#ifndef __giftSelectFeature_H
#define __giftSelectFeature_H

//ITK includes


//GIFT includes
#include "giftWeightGenerator.h"

 namespace gift {

/**
 * \class SelectFeature
 * \brief Generators weight maps by setting the weight to be 1.0 for the 
 *        the selected level/band image, and 0.0 for others. 
 *
 * \ingroup Image Fusion Weight Generators
 */
template <class TImage>
class SelectFeature
    : public WeightGenerator<TImage>
{
public:
    /** Standard class typedefs. */
    typedef SelectFeature                       Self;
    typedef WeightGenerator<TImage>             Superclass;
    typedef itk::SmartPointer<Self>             Pointer;
    typedef itk::SmartPointer<const Self>       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(SelectFeature, WeightGenerator);

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

    /** Sets the feature to select */
    void SetIndexToSelect(unsigned int index)
    {
        this->m_IndexToSelect = index;      
    }
    
protected:
    SelectFeature();
    ~SelectFeature(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

private:
    SelectFeature(const Self&);     //purposely not implemented
    void operator=(const Self&);    //purposely not implemented

    unsigned int m_IndexToSelect;
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftSelectFeature.txx"
#endif

#endif
