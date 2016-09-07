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

/*=========================================================================

Program: Insight Segmentation & Registration Toolkit
Module: $RCSfile: $
Language: C++
Date: $Date: 2007-08-31 22:17:25 +0200 (Fri, 31 Aug 2007) $
Version: $Revision: 2 $
Author: Gavin Baker <gavinb@cs.mu.oz.au>

Copyright (c) 2004 Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

=========================================================================*/

#ifndef rtkAdditiveGaussianNoiseImageFilter_hxx
#define rtkAdditiveGaussianNoiseImageFilter_hxx

#include "rtkAdditiveGaussianNoiseImageFilter.h"

namespace rtk
{

template <class TInputImage>
AdditiveGaussianNoiseImageFilter<TInputImage>
::AdditiveGaussianNoiseImageFilter()
{
  m_NoiseFilter = NoiseFilterType::New();
  m_NoiseFilter->GetFunctor().SetOutputMinimum( itk::NumericTraits< InputPixelType >::NonpositiveMin() );
  m_NoiseFilter->GetFunctor().SetOutputMaximum( itk::NumericTraits< InputPixelType >::max() );
}


template <class TInputImage>
void
AdditiveGaussianNoiseImageFilter<TInputImage>
::GenerateData()
{
  this->AllocateOutputs();

  // Set the global max number of threads to 1
  // NOTE: This is required because there is a bug with this filter,
  // it appears the NormalVariateGenerate is single threaded only.
  m_NoiseFilter->SetNumberOfThreads(1);

  // Setup grafted pipeline for composite filter
  m_NoiseFilter->SetInput( this->GetInput() );
  m_NoiseFilter->Update();
  this->GraftOutput( m_NoiseFilter->GetOutput() );
}

template <class TInputImage>
void
AdditiveGaussianNoiseImageFilter<TInputImage>
::PrintSelf(std::ostream& os,
itk::Indent indent) const
{
  os
  << indent << "AdditiveGaussianNoiseImageFilter"
  << "\n Mean: " << this->GetMean()
  << "\n StandardDeviation: " << this->GetStandardDeviation()
  << std::endl;
}

} /* namespace rtk */

#endif // rtkAdditiveGaussianNoiseImageFilter_hxx
