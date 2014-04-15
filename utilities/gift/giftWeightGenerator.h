/*=========================================================================

  Program:  GIFT Weight Generator
  Module:   giftWeightGenerator.h
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
#ifndef __giftWeightGenerator_H
#define __giftWeightGenerator_H

//ITK includes

//GIFT includes
#include "giftMultilevelMultibandImageFilter.h"
#include "giftFeatureGenerator.h"

namespace gift {

/**
 * \class WeightGenerator
 * \brief Inputs a number of feature maps and returns a weight map for each
 *        input band image.
 *
 * WeightGenerators are only passed to features that they need. Needed
 * features can be set and retrieved by the AddFeatureToUse() and
 * IsFeatureToBeUsed() methods respectively.
 *
 * This is an abstract class which exposes the SetNumberOfFeatureMaps() and 
 * SetNumberOfWeightMaps() methods. Subclasses must override the GenerateData()
 * method. Each feature map is for an individual band, and therefore this
 * class has NumberOfLevels = 1, NumberOfBands = 1).
 *
 * \ingroup Image Fusion Weight Generator
 */
template <class TImage>
class WeightGenerator
    : public MultilevelMultibandImageFilter<TImage>
{
public:
    /** Standard class typedefs. */
    typedef WeightGenerator                         Self;
    typedef MultilevelMultibandImageFilter<TImage>  Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Run-time type information (and related methods). */
    itkTypeMacro(WeightGenerator, MultilevelMultibandImageFilter);

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
    
    /** FeatureGenerator typedefs */
    //NOTE: This declaration is copied from ImageFusionFilter.h (it can't be included
    //      due to circular include issues...)
    typedef FeatureGenerator<TImage>                FeatureGeneratorType;
    typedef typename FeatureGeneratorType::Pointer  FeatureGeneratorPointer;
    typedef std::vector<FeatureGeneratorPointer>    FeatureGeneratorContainer;
    
    /** Sets the number of input feature maps. */
    virtual void SetNumberOfFeatureMaps(unsigned int features)
    {
        Superclass::SetNumberOfInputImages(features);
    }

    /** Sets the number of output weight maps. */
    virtual void SetNumberOfWeightMaps(unsigned int weights)
    {
        Superclass::SetNumberOfOutputImages(weights);
    }

    /** Gets the number of input feature maps. */
    virtual unsigned int GetNumberOfFeatureMaps()
    {
        return Superclass::GetNumberOfInputImages();
    }

    /** Sets the number of output weight maps. */
    virtual unsigned int GetNumberOfWeightMaps()
    {
        return Superclass::GetNumberOfOutputImages();
    }

    /** Overrides the expected number of input levels */
    unsigned int GetNumberOfInputLevels()
    {
        return 1;
    }

    /** Overrides the expected number of output levels */
    unsigned int GetNumberOfOutputLevels()
    {
        return 1;
    }

    /** Overrides the expected number of input bands */
    unsigned int GetNumberOfInputBands()
    {
        return 1;
    }

    /** Overrides the expected number of output bands */
    unsigned int GetNumberOfOutputBands()
    {
        return 1;
    }

    /** Add a feature to be used by this WeightGenerator */
    void AddFeatureToUse(FeatureGeneratorPointer feature)
    {
        this->m_FeaturesToUse.push_back(feature);   
    }

    /** Returns the number of features needed by this WeightGenerator */
    unsigned int GetNumberOfUsedFeatures()
    {
        return this->m_FeaturesToUse.size();
    }

    /** Returns if a given feature is to be used by this WeightGenerator */
    bool IsFeatureToBeUsed(FeatureGeneratorPointer feature);


protected:
    WeightGenerator();
    ~WeightGenerator(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData() = 0;

private:
    WeightGenerator(const Self&);       //purposely not implemented
    void operator=(const Self&);        //purposely not implemented

    FeatureGeneratorContainer m_FeaturesToUse;
};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftWeightGenerator.txx"
#endif

#endif
