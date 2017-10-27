/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkDeconstructSoftThresholdReconstructImageFilter_h
#define rtkDeconstructSoftThresholdReconstructImageFilter_h

//ITK includes
#include "itkMacro.h"
#include "itkProgressReporter.h"

//rtk includes
#include "rtkDeconstructImageFilter.h"
#include "rtkReconstructImageFilter.h"
#include "rtkSoftThresholdImageFilter.h"

namespace rtk {

/**
 * \class DeconstructSoftThresholdReconstructImageFilter
 * \brief Deconstructs an image, soft thresholds its wavelets coefficients,
 * then reconstructs
 *
 * This filter is inspired from Dan Mueller's GIFT package
 * http://www.insight-journal.org/browse/publication/103
 *
 * \author Cyril Mory
 */
template <class TImage>
class DeconstructSoftThresholdReconstructImageFilter
    : public itk::ImageToImageFilter<TImage,TImage>
{
public:
    /** Standard class typedefs. */
    typedef DeconstructSoftThresholdReconstructImageFilter                   Self;
    typedef itk::ImageToImageFilter<TImage,TImage>  Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(DeconstructSoftThresholdReconstructImageFilter, ImageToImageFilter)

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

    /** Define the types of subfilters */
    typedef rtk::DeconstructImageFilter<InputImageType>       DeconstructFilterType;
    typedef rtk::ReconstructImageFilter<InputImageType>       ReconstructFilterType;
    typedef rtk::SoftThresholdImageFilter<InputImageType, InputImageType>   SoftThresholdFilterType;

    /** Set the number of levels of the deconstruction and reconstruction */
    void SetNumberOfLevels(unsigned int levels);

    /** Sets the order of the Daubechies wavelet used to deconstruct/reconstruct the image pyramid */
    itkGetMacro(Order, unsigned int)
    itkSetMacro(Order, unsigned int)

    /** Sets the threshold used in soft thresholding */
    itkGetMacro(Threshold, float)
    itkSetMacro(Threshold, float)

protected:
    DeconstructSoftThresholdReconstructImageFilter();
    ~DeconstructSoftThresholdReconstructImageFilter() {}
    void PrintSelf(std::ostream&os, itk::Indent indent) const ITK_OVERRIDE;

    /** Generate the output data. */
    void GenerateData() ITK_OVERRIDE;

    /** Compute the information on output's size and index */
    void GenerateOutputInformation() ITK_OVERRIDE;

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

private:
    DeconstructSoftThresholdReconstructImageFilter(const Self&);     //purposely not implemented
    void operator=(const Self&);            //purposely not implemented

    unsigned int    m_Order;
    float           m_Threshold;
    bool            m_PipelineConstructed;

    typename DeconstructFilterType::Pointer                 m_DeconstructionFilter;
    typename ReconstructFilterType::Pointer                 m_ReconstructionFilter;
    std::vector<typename SoftThresholdFilterType::Pointer>  m_SoftTresholdFilters; //Holds an array of soft threshold filters

};

}// namespace rtk

//Include CXX
#ifndef rtk_MANUAL_INSTANTIATION
#include "rtkDeconstructSoftThresholdReconstructImageFilter.hxx"
#endif

#endif
