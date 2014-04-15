/*=========================================================================

  Program:  GIFT QMF Wavelet Image Filter
  Module:   giftQmfWaveletImageFilter.txx
  Language: C++
  Date:     2005/11/16
  Version:  0.1
  Author:   Dan Mueller [d.mueller@qut.edu.au]

  Copyright (c) 2005 Queensland University of Technology. All rights reserved.
  See giftCopyright.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __giftQmfWaveletImageFilter_TXX
#define __giftQmfWaveletImageFilter_TXX

//GIFT Includes
#include "giftQmfWaveletImageFilter.h"
#include "giftUpsampleImageFilter.h"
#include "giftDownsampleImageFilter.h"

//ITK includes
#include "itkConstantBoundaryCondition.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "itkPeriodicBoundaryCondition.h"
#include "itkNeighborhoodOperatorImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkImageBase.h"

namespace gift
{

/////////////////////////////////////////////////////////
//Constructor()
template <class TImage, class TWavelet>
QmfWaveletImageFilter<TImage, TWavelet>
::QmfWaveletImageFilter()
{
    //Init images
    this->SetNumberOfInputImages(this->GetNumberOfInputImages());
    this->SetNumberOfOutputImages(this->GetNumberOfOutputImages());

    //Init levels - default is 1
    this->SetNumberOfLevels(1);
}


/////////////////////////////////////////////////////////
//PrintSelf()
template <class TImage, class TWavelet>
void
QmfWaveletImageFilter<TImage, TWavelet>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
    Superclass::PrintSelf(os, indent);
}



/////////////////////////////////////////////////////////
//AddFiltersForDimension()
template <class TImage, class TWavelet>
void
QmfWaveletImageFilter<TImage, TWavelet>
::AddFiltersForDimension(unsigned int idim, std::vector<itk::DataObject::Pointer>& inputs, itk::ProgressReporter &progress)
{
    //Local typedefs and declarations
    //typedef itk::PeriodicBoundaryCondition<TImage> BoundaryConditionType;
    //typedef itk::ZeroFluxNeumannBoundaryCondition<TImage> BoundaryConditionType;
    typedef itk::ConstantBoundaryCondition<TImage> BoundaryConditionType;

    typedef itk::NeighborhoodOperatorImageFilter<TImage,
                                                 TImage> ConvolutionFilterType;
    typedef DownsampleImageFilter<TImage,
                                  TImage> DownsampleFilterType;
    typedef UpsampleImageFilter<TImage,
                                TImage> UpsampleFilterType;
    typedef itk::AddImageFilter<TImage, 
                                TImage, 
                                TImage> AddFilterType;  
    BoundaryConditionType boundaryCondition;
    
    //Create a copy of the inputs vector
    //NOTE: inputs is what we'll write to
    //NOTE: inputsRead is what we'll read from
    std::vector<itk::DataObject::Pointer> inputRead(inputs);
    inputs.clear();

    //Switch between deconstruction and reconstruction
    if (this->IsDeconstruction())
    {
        for (unsigned int indexInputs=0; indexInputs<inputRead.size(); indexInputs++)
        {
            //LOW-PASS
            //---------------------------------------------
            //Setup Wavelet Operator
            TWavelet lowpassWavelet(this->m_Wavelet.GetName());
            lowpassWavelet.SetDirection(idim);
            lowpassWavelet.SetLowpassDeconstruction();
            lowpassWavelet.CreateDirectional();

            //Add convolution filter
            typename ConvolutionFilterType::Pointer convolutionFilter0 = ConvolutionFilterType::New();
            convolutionFilter0->SetOperator(lowpassWavelet);
            convolutionFilter0->OverrideBoundaryCondition(&boundaryCondition);
            itk::DataObject *ptrConvolutionInput0 = inputRead[indexInputs];
            typename itk::ImageToImageFilter<TImage, TImage>::InputImageType* convolutionInput0 
               = dynamic_cast<typename itk::ImageToImageFilter<TImage, TImage>::InputImageType*>(ptrConvolutionInput0);
            convolutionFilter0->SetInput(convolutionInput0);
            convolutionFilter0->Update();
            progress.CompletedPixel();

            //Add downsample filter
            typename DownsampleFilterType::Pointer downsampleFilter0 = DownsampleFilterType::New();
            downsampleFilter0->SetInput(convolutionFilter0->GetOutput());
            downsampleFilter0->SetFactor(idim, 2);
            downsampleFilter0->Update();
            progress.CompletedPixel();

            //Add to output
            typename gift::DownsampleImageFilter<TImage, TImage>::OutputImageType* output0 = downsampleFilter0->GetOutput();
            inputs.push_back(output0);

            //HIGH-PASS
            //---------------------------------------------
            //Setup Wavelet Operator
            TWavelet highpassWavelet(this->m_Wavelet.GetName());
            highpassWavelet.SetDirection(idim);
            highpassWavelet.SetHighpassDeconstruction();
            highpassWavelet.CreateDirectional();

            //Add convolution filter
            typename ConvolutionFilterType::Pointer convolutionFilter1 = ConvolutionFilterType::New();
            convolutionFilter1->SetOperator(highpassWavelet);
            convolutionFilter1->OverrideBoundaryCondition(&boundaryCondition);
            itk::DataObject *ptrConvolutionInput1 = inputRead[indexInputs];
            typename itk::ImageToImageFilter<TImage, TImage>::InputImageType* convolutionInput1
              = dynamic_cast<typename itk::ImageToImageFilter<TImage, TImage>::InputImageType*>(ptrConvolutionInput1);
            convolutionFilter1->SetInput(convolutionInput1);
            convolutionFilter1->Update();
            progress.CompletedPixel();

            //Add downsample filter
            typename DownsampleFilterType::Pointer downsampleFilter1 = DownsampleFilterType::New();
            downsampleFilter1->SetInput(convolutionFilter1->GetOutput());
            downsampleFilter1->SetFactor(idim, 2);
            downsampleFilter1->Update();
            progress.CompletedPixel();

            //Add to output
            typename gift::DownsampleImageFilter<TImage, TImage>::OutputImageType* output1 = downsampleFilter1->GetOutput();
            inputs.push_back(output1);
        }
    }
    else if (this->IsReconstruction())
    {
        //Do for each input (in reverse order)
        for (unsigned int indexInputs=0; indexInputs<inputRead.size(); indexInputs+=2)
        {
            //LOW-PASS
            //---------------------------------------------
            //Setup Wavelet Operator
            TWavelet lowpassWavelet(this->m_Wavelet.GetName());
            lowpassWavelet.SetDirection(idim);
            lowpassWavelet.SetLowpassReconstruction();
            lowpassWavelet.CreateDirectional();

            //Add upsample filter
            typename UpsampleFilterType::Pointer upsampleFilter0 = UpsampleFilterType::New();
            itk::DataObject *ptrUpsampleInput0 = inputRead[indexInputs];
            typename itk::ImageToImageFilter<TImage, TImage>::InputImageType* upsampleInput0 
              = dynamic_cast<typename itk::ImageToImageFilter<TImage, TImage>::InputImageType*>(ptrUpsampleInput0);
            upsampleFilter0->SetInput(upsampleInput0);
            upsampleFilter0->SetFactor(idim, 2);
            upsampleFilter0->Update();
            progress.CompletedPixel();

            //Add convolution filter
            typename ConvolutionFilterType::Pointer convolutionFilter0 = ConvolutionFilterType::New();
            convolutionFilter0->SetOperator(lowpassWavelet);
            convolutionFilter0->OverrideBoundaryCondition(&boundaryCondition);
            convolutionFilter0->SetInput(upsampleFilter0->GetOutput());
            convolutionFilter0->Update();
            progress.CompletedPixel();

            //HIGH-PASS
            //---------------------------------------------
            //Setup Wavelet Operator
            TWavelet highpassWavelet(this->m_Wavelet.GetName());
            highpassWavelet.SetDirection(idim);
            highpassWavelet.SetHighpassReconstruction();
            highpassWavelet.CreateDirectional();

            //Add upsample filter
            typename UpsampleFilterType::Pointer upsampleFilter1 = UpsampleFilterType::New();
            itk::DataObject *ptrUpsampleInput1 = inputRead[indexInputs+1];
            typename itk::ImageToImageFilter<TImage, TImage>::InputImageType* upsampleInput1 
              = dynamic_cast<typename itk::ImageToImageFilter<TImage, TImage>::InputImageType*>(ptrUpsampleInput1);
            upsampleFilter1->SetInput(upsampleInput1);
            upsampleFilter1->SetFactor(idim, 2);
            upsampleFilter1->Update();
            progress.CompletedPixel();

            //Add convolution filter
            typename ConvolutionFilterType::Pointer convolutionFilter1 = ConvolutionFilterType::New();
            convolutionFilter1->SetOperator(highpassWavelet);
            convolutionFilter1->OverrideBoundaryCondition(&boundaryCondition);
            convolutionFilter1->SetInput(upsampleFilter1->GetOutput());
            convolutionFilter1->Update();
            progress.CompletedPixel();

            //Add and send to output
            //---------------------------------------------
            typename AddFilterType::Pointer addFilter = AddFilterType::New();
            addFilter->SetInput1(convolutionFilter0->GetOutput()); //LP
            addFilter->SetInput2(convolutionFilter1->GetOutput()); //HP
            addFilter->Update();
            progress.CompletedPixel();

            typename itk::AddImageFilter<TImage, TImage, TImage>::OutputImageType* outputAdd = addFilter->GetOutput();
            inputs.push_back(outputAdd);        
        } 
    }
}

                  
/////////////////////////////////////////////////////////
//GenerateData()
template <class TImage, class TWavelet>
void
QmfWaveletImageFilter<TImage, TWavelet>
::GenerateData()
{
    //Get number of input and output images
    unsigned int numberOfInputs = this->GetNumberOfRequiredInputs();
    unsigned int numberOfOutputs = this->GetNumberOfRequiredOutputs();

    //Allocate memory for outputs
    for (unsigned int idx = 0; idx < numberOfOutputs; idx++)
    {
        OutputImagePointer outputPtr = this->GetOutput(idx);
        outputPtr->SetBufferedRegion(outputPtr->GetRequestedRegion());
        outputPtr->Allocate();
    }

    //TODO:
    //Make work for multiple images
    unsigned int iimage = 0;

    //Setup output container
    std::vector<itk::DataObject::Pointer> outputs;
    outputs.resize(numberOfOutputs);

    //Setup default input container
    std::vector<itk::DataObject::Pointer> inputs;
    for (unsigned int idx = 0; idx < numberOfInputs; idx++)
    {
        inputs.push_back(const_cast<InputImageType*>(static_cast<const InputImageType*>(this->GetInput(idx))));
    }

    //Setup ProgressReporter
    unsigned int numberOfParts = 0;
    for (unsigned int idim = 0; idim < ImageDimension; idim++)
    {
        if (this->IsDeconstruction())
        {
            numberOfParts += (unsigned int)pow(2.0, (int)idim+2);
        }
        else if (this->IsReconstruction())
        {
            numberOfParts += ((unsigned int)pow(2.0, (int)idim+2) + (unsigned int)pow(2.0, (int)idim));   
        }
    }
    numberOfParts *= this->GetNumberOfLevels();
    itk::ProgressReporter progress(this, 0, numberOfParts);
    
    //Add filters for each input level
    for (unsigned int ilevel=0; ilevel < this->GetNumberOfLevels(); ilevel++)
    {
        //Add filters for each dimension
        if (this->IsDeconstruction())
        {
            //Add filters
            for (unsigned int idim = 0; idim < ImageDimension; idim++)
            {
                this->AddFiltersForDimension(idim, inputs, progress);
            }

            //Set outputs
            //NOTE: band=0 is used to calculate the next level
            for (unsigned int iband=0; iband<this->GetNumberOfOutputBands(); iband++)
            {
                unsigned int outputIndex = this->ConvertOutputImageLevelBandToIndex(iimage, ilevel, iband);
                outputs[outputIndex] = inputs[iband];
            }

            //Set inputs for next level
            inputs.clear();
            inputs.push_back(outputs[this->ConvertOutputImageLevelBandToIndex(iimage, ilevel, 0)]);
        }
        else if (this->IsReconstruction())
        {
            //Convert ilevel to count backwards
            //NOTE: The inputs are arranged with smaller inputs in larger indices.
            //NOTE: Therefore we need to get the smaller inputs first, and then work
            //      our way back to the bigger images (smaller indices)
            //SUMMARY: Larger levels = smaller images
            unsigned int ilevelBackwards = this->GetNumberOfLevels() - ilevel - 1;

            //Get subset of inputs
            std::vector<itk::DataObject::Pointer> inputsSubset;
            inputsSubset.resize(this->GetNumberOfInputBands());
            for (unsigned int iband=0; iband<this->GetNumberOfInputBands(); iband++)
            {
                unsigned int inputIndex = this->ConvertInputImageLevelBandToIndex(iimage, ilevelBackwards, iband);
                inputsSubset[iband] = inputs[inputIndex];
            }

            //Replace first band image with previous level output
            //NOTE: If this is the first level, just use the given input
            if (ilevel > 0)
            {
                inputsSubset[0] = outputs[0];
            }

            //Add filters
            for (int idim = (ImageDimension-1); idim >=0; idim--)
            {
                this->AddFiltersForDimension(idim, inputsSubset, progress);
            }

            //Save previous level output
            outputs.clear();
            outputs.push_back(inputsSubset[0]);
        }
    }
    
    //Graft to output
    for (unsigned int index=0; index<outputs.size(); index++)
    {
        itk::DataObject *ptrOutputToGraft = outputs[index];
        typename itk::ImageToImageFilter<TImage, TImage>::OutputImageType* outputToGraft 
          = dynamic_cast<typename itk::ImageToImageFilter<TImage, TImage>::OutputImageType*>(ptrOutputToGraft);
        this->GraftNthOutput(index, outputToGraft); 
    }
}


/////////////////////////////////////////////////////////
//GenerateInputRequestedRegion()
template <class TImage, class TWavelet>
void
QmfWaveletImageFilter<TImage, TWavelet>
::GenerateInputRequestedRegion()
{
    //Call the superclass' implementation of this method
    //Superclass::GenerateInputRequestedRegion();

    for (unsigned int idx = 0; idx < this->GetNumberOfInputs(); ++idx)
    {
        //Get the pointer to the current input
        InputImagePointer inputPtr = const_cast<InputImageType *>(this->GetInput(idx));
        if (!inputPtr)
        {
            itkExceptionMacro( << "Input has not been set." );
        }

        //Some useful typedefs
        typedef typename OutputImageType::SizeType    SizeType;
        typedef typename SizeType::SizeValueType      SizeValueType;
        typedef typename OutputImageType::IndexType   IndexType;
        typedef typename IndexType::IndexValueType    IndexValueType;
        typedef typename OutputImageType::RegionType  RegionType;

        //Compute the output difference factor
        unsigned int image = 0;
        unsigned int level = 0;
        unsigned int band = 0;
        this->ConvertOutputIndexToImageLevelBand(idx, image, level, band);
        unsigned int factor = (unsigned int)pow(2.0,(int)(level+1));
        
        //Compute baseIndex and baseSize (from first output)
        SizeType baseSize = this->GetOutput(0)->GetRequestedRegion().GetSize();
        IndexType baseIndex = this->GetOutput(0)->GetRequestedRegion().GetIndex();
        RegionType baseRegion;

        for (unsigned int idim = 0; idim < ImageDimension; idim++)
        {
            //Switch between multiplying or dividing depending on if we are
            //deconstructing or reconstructing
            if (this->IsReconstruction())
            {
                baseSize[idim]  = (long)ceil((double)baseSize[idim]   / (double)factor);
                baseIndex[idim] = (long)floor((double)baseIndex[idim] / (double)factor);
            }
            else if (this->IsDeconstruction())
            {
                baseIndex[idim] *= factor;
                baseSize[idim] *= factor;
            }
        }
        baseRegion.SetIndex(baseIndex);
        baseRegion.SetSize(baseSize);

        //Pad region by wavelet radius
//        unsigned long radius[ImageDimension];
        IndexValueType radius[ImageDimension];
        RegionType inputRequestedRegion = baseRegion;

        for (unsigned int idim = 0; idim < TImage::ImageDimension; idim++)
        {
            this->m_Wavelet.SetDirection(idim);
            this->m_Wavelet.CreateDirectional();
            radius[idim] = this->m_Wavelet.GetRadius()[idim];
        }

        inputRequestedRegion.PadByRadius(radius);

        //Make sure the requested region is within the largest possible
        inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion());

        //Set the input requested region
        inputPtr->SetRequestedRegion(inputRequestedRegion);
    }
}


/////////////////////////////////////////////////////////
//GenerateOutputInformation()
template <class TImage, class TWavelet>
void
QmfWaveletImageFilter<TImage, TWavelet>
::GenerateOutputInformation()
{
    //Call the superclass's implementation of this method
    Superclass::GenerateOutputInformation();

   //NOTE: if IsReconstruction - nothing more to do...
    
    if (this->IsDeconstruction())
    {
        //Get number of required outputs
        unsigned int numberOfOutputs = this->CalculateNumberOfOutputs();
        
        typedef typename OutputImageType::SizeType SizeType;
        typedef typename SizeType::SizeValueType   SizeValueType;
        typedef typename OutputImageType::IndexType IndexType;
        typedef typename IndexType::IndexValueType  IndexValueType;

        typename OutputImageType::SpacingType outputSpacing;
        SizeType        outputSize;
        IndexType       outputStartIndex;
      
        //We need to compute the output spacing, the output image size,
        //and the output image start index
        for (unsigned int index = 0; index < numberOfOutputs; index++)
        {
            //Compute the output different factor
            unsigned int image = 0;
            unsigned int level = 0;
            unsigned int band = 0;
            this->ConvertOutputIndexToImageLevelBand(index, image, level, band);
            double factor = pow(2.0, (double)(level+1));
            
            //Get input pointer
            InputImageConstPointer inputPtr = this->GetInput(0);

            if (!inputPtr)
            {
                itkExceptionMacro( << "Input has not been set" );
            }

            const typename InputImageType::SpacingType& inputSpacing = inputPtr->GetSpacing();
            const typename InputImageType::SizeType& inputSize = inputPtr->GetLargestPossibleRegion().GetSize();
            const typename InputImageType::IndexType& inputStartIndex = inputPtr->GetLargestPossibleRegion().GetIndex();

            //Get output pointer
            OutputImagePointer outputPtr = this->GetOutput(index);
            if (!outputPtr) { continue; }

            //Get spacing and size foreach dimension
            for (unsigned int idim = 0; idim < TImage::ImageDimension; idim++)
            {
                outputSpacing[idim] = inputSpacing[idim] * factor;
                outputSize[idim] = static_cast<SizeValueType>(floor(static_cast<double>(inputSize[idim]) / factor));
                
                if (outputSize[idim] < 1) { outputSize[idim] = 1; }
                outputStartIndex[idim] = static_cast<IndexValueType>(ceil(static_cast<double>(inputStartIndex[idim]) / factor));
            }
      
            //Set size and index for region
            typename OutputImageType::RegionType outputLargestPossibleRegion;
            outputLargestPossibleRegion.SetSize(outputSize);
            outputLargestPossibleRegion.SetIndex(outputStartIndex);

            //Set region and spacing
            outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
            outputPtr->SetSpacing(outputSpacing);
            
        }
    }
}


/////////////////////////////////////////////////////////
//GenerateOutputRequestedRegion()
template <class TImage, class TWavelet>
void
QmfWaveletImageFilter<TImage, TWavelet>
::GenerateOutputRequestedRegion(itk::DataObject *output)
{
    //Call the superclass's implementation of this method
    Superclass::GenerateOutputInformation();

      //NOTE: if IsReconstruction - nothing more to do...
    
    if (this->IsDeconstruction())
    {
        for (unsigned int index=0; index < this->GetNumberOfOutputs(); ++index)
        {
            itk::ImageBase<ImageDimension> *givenData;  
            itk::ImageBase<ImageDimension> *currentData;  
            givenData = dynamic_cast<itk::ImageBase<ImageDimension>*>(output);
            currentData = dynamic_cast<itk::ImageBase<ImageDimension>*>(this->GetOutput(index));
            
            if (currentData && givenData && givenData != currentData)
            {
                typename itk::ImageBase<ImageDimension>::RegionType givenRegion = givenData->GetRequestedRegion();
                typename itk::ImageBase<ImageDimension>::RegionType currentRegion = currentData->GetRequestedRegion();
                
                //Get image/level/band and difference factor
                unsigned int image = 0;
                unsigned int level = 0;
                unsigned int band = 0;
                this->ConvertOutputIndexToImageLevelBand(index, image, level, band);
                double factor = pow(2.0, (double)(level));

                //Convert given output requested region by factor
                for (unsigned int idim = 0; idim < ImageDimension; idim++)
                {
                    currentRegion.SetSize(idim, givenRegion.GetSize(idim) / factor);
                }
                
                currentData->SetRequestedRegion(currentRegion);
            }//end if
        }//end for 
    }//end if (IsDeconstruction)
}


}// end namespace gift

#endif
