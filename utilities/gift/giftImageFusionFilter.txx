/*=========================================================================

  Program:  GIFT Image Fusion Filter
  Module:   giftImageFusionFilter.txx
  Language: C++
  Date:     2006/06/07
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftImageFusionFilter_TXX
#define __giftImageFusionFilter_TXX

//GIFT Includes
#include "giftImageFusionFilter.h"

//ITK includes
#include "itkNumericTraits.h"
#include "itkThresholdImageFilter.h"
#include "itkCastImageFilter.h"

namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TInternalImage, class TOutputImage>
ImageFusionFilter<TInternalImage, TOutputImage>
::ImageFusionFilter()
{
    //this->m_FeatureGenerators.resize(0);
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TInternalImage, class TOutputImage>
void
ImageFusionFilter<TInternalImage, TOutputImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}



/////////////////////////////////////////////////////////
//OverrideWeightGenerator()
template <class TInternalImage, class TOutputImage>
void
ImageFusionFilter<TInternalImage, TOutputImage>
::OverrideWeightGenerator(WeightGeneratorPointer weightGenerator,
                          unsigned int level,
                          unsigned int band)
{
    //Convert level/band to key
    WeightGeneratorKey key(level, band);
    
    //Insert entry into map
    WeightGeneratorEntry entry(key, weightGenerator);
    this->m_OverridenWeightGenerators.insert(entry);
}


/////////////////////////////////////////////////////////
//GenerateData()
template <class TInternalImage, class TOutputImage>
void
ImageFusionFilter<TInternalImage, TOutputImage>
::GenerateData()
{
    //Get number of images
    unsigned int numberOfImagesToFuse = this->GetNumberOfInputs();

    //===================================================
    //Deconstruct inputs ================================
    //Declare deconstructed output storage
    std::vector<itk::DataObject::Pointer> deconstructed;

    //Setup Multiscale Method
    this->m_MultiscaleMethod->SetDeconstruction();

    //Do for each input
    for (unsigned int idxInput=0; idxInput<numberOfImagesToFuse; idxInput++)
    {   
        //Get input
        itk::DataObject* ptrInput = const_cast<InputImageType*>(static_cast<const InputImageType*>(this->GetInput(idxInput)));
        typename MultiscaleMethodType::InputImageType* input 
          = dynamic_cast<typename MultiscaleMethodType::InputImageType*>(ptrInput);
        
        //Deconstruct
        this->m_MultiscaleMethod->SetInput(0, input);
        this->m_MultiscaleMethod->Update();

        //Save output to deconstructed
        for (unsigned int idxDeconstructed=0; idxDeconstructed<this->m_MultiscaleMethod->GetNumberOfOutputs(); idxDeconstructed++)
        {
            typename MultiscaleMethodType::OutputImageType* deconstructedOutput = this->m_MultiscaleMethod->GetOutput(idxDeconstructed);
            deconstructed.push_back(deconstructedOutput);
            //NOTE: We need to disconnect output image from the pipeline so we 
            //      can reuse this->m_MultiscaleMethod->Outputs()
            deconstructedOutput->DisconnectPipeline(); 
        }
    }   

    //===================================================
    //Apply feature generators ==========================
    unsigned int numberOfFeatureMaps = 0;

    //Do foreach feature generator
    typename FeatureGeneratorContainer::iterator itFeatureGenerator;
    for (itFeatureGenerator  = m_FeatureGenerators.begin(); 
         itFeatureGenerator != m_FeatureGenerators.end(); 
         itFeatureGenerator++)
    {
        //Get current FeatureGenerator
        FeatureGeneratorPointer currentFeatureGenerator = *itFeatureGenerator;

        //Set the number of input images, levels, bands for each FeatureGenerator.
        //NOTE: the feature generator may ignore these if specific overrides
        //      have been implemented (eg. features that take the original 
        //      single-level input images)
        //NOTE: feature generators are responsible for setting the output images,
        //      levels, and bands after the input details have been set.
        currentFeatureGenerator->SetNumberOfInputImages(numberOfImagesToFuse);
        currentFeatureGenerator->SetNumberOfInputLevels(this->m_MultiscaleMethod->GetNumberOfOutputLevels());
        currentFeatureGenerator->SetNumberOfInputBands(this->m_MultiscaleMethod->GetNumberOfOutputBands());

        //Determine if we pass 1. the orginial single-level input images  -OR-
        //                     2. the deconstructed multi-level input images
        if (currentFeatureGenerator->GetNumberOfInputLevels() == 0)
        {
            //1. Pass all single-level input images
            for (unsigned int idxInput=0; idxInput<numberOfImagesToFuse; idxInput++)
            {               
                //Get input
                itk::DataObject* ptrInput = const_cast<InputImageType*>(static_cast<const InputImageType*>(this->GetInput(idxInput)));
                typename FeatureGeneratorType::InputImageType* input 
                  = dynamic_cast<typename FeatureGeneratorType::InputImageType*>(ptrInput);

                //Add to feature generator
                currentFeatureGenerator->SetInput(idxInput, input);

            }//end foreach input
        }
        else
        {
            //2. Pass all deconstructed multi-level input images
            for (unsigned int idxInput=0; idxInput<deconstructed.size(); idxInput++)
            {
                //Get input
                itk::DataObject* ptrInput = deconstructed[idxInput];
                typename FeatureGeneratorType::InputImageType* input 
                  = dynamic_cast<typename FeatureGeneratorType::InputImageType*>(ptrInput);

                //Add to feature generator
                currentFeatureGenerator->SetInput(idxInput, input);

            }//end foreach input
        }

        //Update Feature Generator
        currentFeatureGenerator->Update();
        numberOfFeatureMaps++;
    }


    //===================================================
    //Setup weight combiner =============================                      
    this->m_WeightCombiner->SetNumberOfImages(numberOfImagesToFuse);
    this->m_WeightCombiner->SetNumberOfLevels(this->m_MultiscaleMethod->GetNumberOfOutputLevels());
    this->m_WeightCombiner->SetNumberOfBands(this->m_MultiscaleMethod->GetNumberOfOutputBands());

    //Add deconstructed inputs to weight combiner
    for (unsigned int idxInput=0; idxInput<deconstructed.size(); idxInput++)
    {
        //Get input
        itk::DataObject* ptrInput = deconstructed[idxInput];
        typename WeightCombinerType::InputImageType* input 
          = dynamic_cast<typename WeightCombinerType::InputImageType*>(ptrInput);

        //Add to weight combiner
        m_WeightCombiner->SetInput(idxInput, input);
    }//end foreach input


    //===================================================
    //Apply weight generators ===========================
    //NOTE: Weight generators work on individual band images.
    
    //foreach level
    for (unsigned int level=0; level<this->m_MultiscaleMethod->GetNumberOfOutputLevels(); level++)
    {           
        //Declare weight generator
        WeightGeneratorPointer currentWeightGenerator;
        unsigned int weightGeneratorInputCount = 0;
        
        //foreach band
        for (unsigned int band=0; band<this->m_MultiscaleMethod->GetNumberOfOutputBands(); band++)
        {
            //Look if there is an overriden weight generator
            WeightGeneratorKey key(level, band);
            typename WeightGeneratorContainer::const_iterator wGenIt = this->m_OverridenWeightGenerators.find(key);

            if (wGenIt == this->m_OverridenWeightGenerators.end())
            {
                //There is no override for this level/band - use default
                currentWeightGenerator = m_DefaultWeightGenerator;
            }
            else
            {
                //Found an override - use it
                currentWeightGenerator = wGenIt->second;
            }

            //Setup weight generator
            //TODO: weight generators should know how many feature maps
            currentWeightGenerator->SetNumberOfFeatureMaps(numberOfImagesToFuse*currentWeightGenerator->GetNumberOfUsedFeatures());
            currentWeightGenerator->SetNumberOfWeightMaps(numberOfImagesToFuse);
            
            //foreach image
            for (unsigned int image=0; image<numberOfImagesToFuse; image++)
            {               
                //foreach feature
                for (itFeatureGenerator = m_FeatureGenerators.begin(); 
                    itFeatureGenerator != m_FeatureGenerators.end(); 
                    itFeatureGenerator++)
                {
                    //Get current FeatureGenerator
                    FeatureGeneratorPointer currentFeatureGenerator = *itFeatureGenerator;

                    //Determine if this feature has valid level/band image(s)
                    //TODO: Think about features that do not have a specific band image
                    //      eg. when featureGenerator->NumberOfOutputBands() are not the same... 
                    //TODO: relax this test...because at the moment only features that
                    //      output results the same "shape" as the MultiscaleMethod are 
                    //      being passed to the weight combiner...
                    bool ignoreCurrentFeature  = false;
                    if (currentFeatureGenerator->GetNumberOfOutputImages() < numberOfImagesToFuse ||
                        currentFeatureGenerator->GetNumberOfOutputLevels() < this->m_MultiscaleMethod->GetNumberOfOutputLevels() ||
                        currentFeatureGenerator->GetNumberOfOutputBands()  < this->m_MultiscaleMethod->GetNumberOfOutputBands())
                    {
                        ignoreCurrentFeature = true;    
                    }
                    else if (currentFeatureGenerator->GetNumberOfOutputLevels() < this->m_MultiscaleMethod->GetNumberOfOutputLevels() ||
                             currentFeatureGenerator->GetNumberOfOutputBands()  < this->m_MultiscaleMethod->GetNumberOfOutputBands())
                    {
                        //TODO: handle situations where the "shape" is not the same due to different images.    
                    }

                    //Determine if this feature is needed by the WeightGenerator
                    if (!currentWeightGenerator->IsFeatureToBeUsed(currentFeatureGenerator))
                    {
                        ignoreCurrentFeature = true;    
                    }

                    //If not being ignored...
                    if (!ignoreCurrentFeature)
                    {
                        itk::DataObject* ptrFeatureMap = currentFeatureGenerator->GetDataObjectOutputByImageLevelBand(image, level, band);
                        typename WeightGeneratorType::InputImageType* featureMap 
                          = dynamic_cast<typename WeightGeneratorType::InputImageType*>(ptrFeatureMap);
                        currentWeightGenerator->SetInput(weightGeneratorInputCount, featureMap);
                        featureMap->DisconnectPipeline();
                        weightGeneratorInputCount++;
                    }

                }//end foreach feature          
            }//end foreach image

            //Update the weight generator and redirect the output to weight combiner
            currentWeightGenerator->Update();
            weightGeneratorInputCount = 0;

            for (unsigned int weightMapIndex=0; weightMapIndex<currentWeightGenerator->GetNumberOfWeightMaps(); weightMapIndex++)
            {
                typename WeightCombinerType::InputImageType* weightMap 
                  = dynamic_cast<typename WeightCombinerType::InputImageType*>(currentWeightGenerator->GetOutput(weightMapIndex));
                //TODO: will weightMapIndex+numberOfImagesToFuse (VVVV) work when NumberOfFeatures>NumberOfImagesToFuse ???
                this->m_WeightCombiner->SetInputByImageLevelBand(weightMapIndex+numberOfImagesToFuse, level, band, weightMap);
                //NOTE: We need to disconnect weightMap from the pipeline so we 
                //      can reuse currentWeightGenerator->Outputs()
                weightMap->DisconnectPipeline();
            }

        }//end foreach band
    }//end foreach level


    //===================================================
    //Update weight combiner ============================
    this->m_WeightCombiner->Update();


    //===================================================
    //Reconstruct output ================================
    this->m_MultiscaleMethod->SetReconstruction();

    for (unsigned int indexRecInput=0; indexRecInput<this->m_WeightCombiner->GetNumberOfOutputs(); indexRecInput++)
    {
        //Add input to multiscale method
        typename WeightCombinerType::OutputImageType* ptrRecInput = this->m_WeightCombiner->GetOutput(indexRecInput);
        typename MultiscaleMethodType::InputImageType* recInput 
          = dynamic_cast<typename MultiscaleMethodType::InputImageType*>(ptrRecInput);
        this->m_MultiscaleMethod->SetInput(indexRecInput, recInput);
    }

    //Update multiscale
    this->m_MultiscaleMethod->Update();


    //===================================================
    //Fix up output and cast ============================
    //Theshold
    typedef itk::ThresholdImageFilter<TInternalImage> ThresholdFilterType;
    typename ThresholdFilterType::Pointer thresholdBelow = ThresholdFilterType::New();
    typename ThresholdFilterType::Pointer thresholdAbove = ThresholdFilterType::New();
    thresholdBelow->ThresholdBelow(itk::NumericTraits<OutputPixelType>::min());
    thresholdBelow->SetOutsideValue(itk::NumericTraits<OutputPixelType>::min());
    thresholdBelow->SetInput(this->m_MultiscaleMethod->GetOutput());
    thresholdAbove->ThresholdAbove(itk::NumericTraits<OutputPixelType>::max());
    thresholdAbove->SetOutsideValue(itk::NumericTraits<OutputPixelType>::max());
    thresholdAbove->SetInput(thresholdBelow->GetOutput());
    
    //Cast
    typedef itk::CastImageFilter<TInternalImage, TOutputImage> CastFilterType;
    typename CastFilterType::Pointer filterCast = CastFilterType::New();
    filterCast->SetInput(thresholdAbove->GetOutput());
    filterCast->Update();


    //===================================================
    //Graft output ======================================
    this->GraftOutput(filterCast->GetOutput());
}

}// end namespace gift

#endif
