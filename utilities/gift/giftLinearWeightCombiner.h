/*=========================================================================

  Program:  GIFT Linear Weight Combiner
  Module:   giftLinearWeightCombiner.h
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
#ifndef __giftLinearWeightCombiner_H
#define __giftLinearWeightCombiner_H

//ITK includes


//GIFT includes
#include "giftWeightCombiner.h"

 namespace gift {

/**
 * \class LinearWeightCombiner
 * \brief Combines the multi-level/multi-band input images in a linear fashion
 *        using the weights for each band image.
 *
 * NOTE: The number of input images (i.e NumberOfInputImages()) should be set
 *       to the number of images to fuse, NOT to the total number of images
 *       (eg. images to fuse + weight images). 
 *
 * \ingroup Image Fusion Combiners
 */
template <class TImage>
class LinearWeightCombiner
    : public WeightCombiner<TImage>
{
public:
    /** Standard class typedefs. */
    typedef LinearWeightCombiner            Self;
    typedef WeightCombiner<TImage>          Superclass;
    typedef itk::SmartPointer<Self>         Pointer;
    typedef itk::SmartPointer<const Self>   ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(LinearWeightCombiner, WeightCombiner);

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
    LinearWeightCombiner();
    ~LinearWeightCombiner(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

private:
    LinearWeightCombiner(const Self&);      //purposely not implemented
    void operator=(const Self&);            //purposely not implemented
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftLinearWeightCombiner.txx"
#endif

#endif
