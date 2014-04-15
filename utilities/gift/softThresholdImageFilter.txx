#ifndef __softThresholdImageFilter_txx
#define __softThresholdImageFilter_txx

#include "softThresholdImageFilter.h"

namespace itk
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
