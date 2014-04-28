#ifndef __softThresholdImageFilter_txx
#define __softThresholdImageFilter_txx

#include "rtkSoftThresholdImageFilter.h"

namespace rtk
{

/**
 *
 */
template <class TInputImage, class TOutputImage>
SoftThresholdImageFilter<TInputImage, TOutputImage>
::SoftThresholdImageFilter()
{
    this->SetThreshold(NumericTraits<InputPixelType>::Zero);
}

/**
 *
 */
template <class TInputImage, class TOutputImage>
void
SoftThresholdImageFilter<TInputImage, TOutputImage>
::SetThreshold(const InputPixelType threshold)
{
    this->GetFunctor().SetThreshold(threshold);
    this->Modified();
}

} // end namespace itk

#endif
