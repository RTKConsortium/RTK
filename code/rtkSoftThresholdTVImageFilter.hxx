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

#ifndef rtkSoftThresholdTVImageFilter_hxx
#define rtkSoftThresholdTVImageFilter_hxx

#include "rtkSoftThresholdTVImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkProgressReporter.h>
#include <itkVectorCastImageFilter.h>

#include "vnl/vnl_math.h"

namespace rtk
{

template< typename TInputImage, typename TRealType, typename TOutputImage >
SoftThresholdTVImageFilter< TInputImage, TRealType, TOutputImage >
::SoftThresholdTVImageFilter()
{
    m_RequestedNumberOfThreads = this->GetNumberOfThreads();
    m_Threshold = 0;
}

template< typename TInputImage, typename TRealType, typename TOutputImage >
void
SoftThresholdTVImageFilter< TInputImage, TRealType, TOutputImage >
::ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                       ThreadIdType threadId)
{
    itk::ImageRegionConstIterator< TInputImage >                     InputIt;
    itk::ImageRegionIterator< TOutputImage >                     OutputIt;

    // Support progress methods/callbacks
    itk::ProgressReporter progress( this, threadId, outputRegionForThread.GetNumberOfPixels() );

    InputIt = itk::ImageRegionConstIterator< TInputImage >(this->GetInput(), outputRegionForThread);
    OutputIt = itk::ImageRegionIterator< TOutputImage >(this->GetOutput(), outputRegionForThread);

    while ( !InputIt.IsAtEnd() )
    {
        float TV = 0;
        for ( unsigned int i = 0; i < ImageDimension; ++i )
        {
            TV += InputIt.Get()[i] * InputIt.Get()[i];
        }
        TV = sqrt(TV); // TV is non-negative
        float ratio;
        float temp = TV - m_Threshold;
        if (temp > 0)
            ratio = temp / TV;
        else
            ratio = 0;

        OutputIt.Set( ratio * InputIt.Get());
        ++InputIt;
        ++OutputIt;
        progress.CompletedPixel();
    }
}

} // end namespace itk

#endif
