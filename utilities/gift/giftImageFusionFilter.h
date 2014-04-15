/*=========================================================================

  Program:  GIFT Image Fusion Filter
  Module:   giftImageFusionFilter.h
  Language: C++
  Date:     2005/11/21
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftImageFusionFilter_H
#define __giftImageFusionFilter_H

//General includes
#include <map>

//ITK includes
#include "itkMacro.h"
#include "itkImageToImageFilter.h"

//GIFT includes
#include "giftMultilevelMultibandImageFilter.h"
#include "giftFeatureGenerator.h"
#include "giftWeightGenerator.h"
#include "giftWeightCombiner.h"

namespace gift {

/** This class operators as the Key for a std::map.
 *  (Note that it must override the < operator). 
*/
class WeightGeneratorKey
{
public:
    /** Constructor */
    WeightGeneratorKey(unsigned int level, unsigned int band)
    {
            m_level = level;
            m_band = band;
    }

    /** Return the level */
    unsigned int GetLevel()
    {
        return m_level;
    }

    /** Return the band */
    unsigned int GetBand()
    {
        return m_band;
    }

    bool operator<(const WeightGeneratorKey &right) const
    {
        //NOTE: Add 1 to the level so that if level=0 we don't get collide
        unsigned int leftValue = (this->m_level+1)*this->m_band;
        unsigned int rightValue = (right.m_level+1)*right.m_band;
        return leftValue < rightValue;
    }

private:
    unsigned int m_level;
    unsigned int m_band;
};


/**
 * \class ImageFusionFilter
 * \brief This filter takes 2 (or more) images and fuses them into a single 
 *        output using a multi-scale approach.
 *
 * An ImageFusionFilter takes a "multi-scale method" (which deconstructs and 
 * reconstructs an image into a multi-level/multi-band image), a number of 
 * "feature generators" (which generate image features used to guide the 
 * fusion process), a "weight generator" (which uses feature maps to 
 * generate a weight map), and a "weight combiner" (to apply the weights to 
 * each band image).
 *
 * The input of this filter is 2 (or more) images with 2 or 3 dimensions with real
 * pixel type, and returns a single image of the same type.
 *
 * The structure of the gift::ImageFusionFilter is based on the following works:
 * [1] G. Piella, "A General Framework for Multiresolution Image Fusion: 
 *     from Pixels to Regions," Center for Mathematics and Computer Science, 
 *     Amsterdam PNA-R0211, 2002.
 * [2] Z. Zhang and R. S. Blum, "A categorization of multiscale-decomposition-based 
 *     image fusion schemes with a performance study for a digital camera 
 *     application," Proceedings of the IEEE, vol. 87, pp. 1315-1326, 1999.
 *
 * \ingroup Image Fusion Filters
 */
template <class TInternalImage, class TOutputImage>
class ImageFusionFilter
    : public itk::ImageToImageFilter<TInternalImage, TOutputImage>
{
public:
    /** Standard class typedefs. */
    typedef ImageFusionFilter                                       Self;
    typedef itk::ImageToImageFilter<TInternalImage,TOutputImage>    Superclass;
    typedef itk::SmartPointer<Self>                                 Pointer;
    typedef itk::SmartPointer<const Self>                           ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(ImageFusionFilter, ImageToImageFilter);

    /** ImageDimension enumeration. */
    itkStaticConstMacro(ImageDimension, unsigned int, TInternalImage::ImageDimension);

    /** Inherit types from Superclass. */
    typedef typename Superclass::InputImageType         InputImageType;
    typedef typename Superclass::OutputImageType        OutputImageType;
    typedef typename Superclass::InputImagePointer      InputImagePointer;
    typedef typename Superclass::OutputImagePointer     OutputImagePointer;
    typedef typename Superclass::InputImageConstPointer InputImageConstPointer;
    typedef typename TInternalImage::PixelType          InternalPixelType;
    typedef typename TOutputImage::PixelType            OutputPixelType;

    /** Type of the MultiscaleMethod */
    typedef MultilevelMultibandImageFilter<TInternalImage>  MultiscaleMethodType;
    typedef typename MultiscaleMethodType::Pointer          MultiscaleMethodPointer;
    
    /** Type of FeatureGenerator */
    //NOTE: These typedefs are mirrored in WeightGenerator
    typedef FeatureGenerator<TInternalImage>        FeatureGeneratorType;
    typedef typename FeatureGeneratorType::Pointer   FeatureGeneratorPointer;
    typedef std::vector<FeatureGeneratorPointer>     FeatureGeneratorContainer;

    /** Types for WeightGenerator */
    typedef WeightGenerator<TInternalImage>             WeightGeneratorType;
    typedef typename WeightGeneratorType::Pointer       WeightGeneratorPointer;
    typedef std::map<WeightGeneratorKey, 
                     WeightGeneratorPointer>            WeightGeneratorContainer;
    typedef std::pair<WeightGeneratorKey,
                      WeightGeneratorPointer>           WeightGeneratorEntry;
    
    /** Type of WeightCombiner */
    typedef WeightCombiner<TInternalImage>          WeightCombinerType;
    typedef typename WeightCombinerType::Pointer    WeightCombinerPointer;

    /** Set the MultiscaleMethod. */
    itkSetObjectMacro( MultiscaleMethod, MultiscaleMethodType );

    /** Set/Get the WeightCombiner. */
    itkSetObjectMacro( WeightCombiner, WeightCombinerType );

    /** Add FeatureGenerator. */
    void AddFeatureGenerator (FeatureGeneratorPointer featureGenerator)
    {
        this->m_FeatureGenerators.push_back(featureGenerator);
    }
    
    /** Set default and override WeightGenerator. */
    itkSetObjectMacro( DefaultWeightGenerator, WeightGeneratorType );
    void OverrideWeightGenerator (WeightGeneratorPointer weightGenerator,
                                  unsigned int level,
                                  unsigned int band);

protected:
    ImageFusionFilter();
    ~ImageFusionFilter(){};
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

private:
    ImageFusionFilter(const Self&);         //purposely not implemented
    void operator=(const Self&);            //purposely not implemented

    //Private storage
    MultiscaleMethodPointer     m_MultiscaleMethod;
    FeatureGeneratorContainer   m_FeatureGenerators;
    WeightGeneratorPointer      m_DefaultWeightGenerator;
    WeightGeneratorContainer    m_OverridenWeightGenerators;
    WeightCombinerPointer       m_WeightCombiner;

};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftImageFusionFilter.txx"
#endif

#endif
