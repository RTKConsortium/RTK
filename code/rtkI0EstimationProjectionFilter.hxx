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

#ifndef rtkI0EstimationProjectionFilter_hxx
#define rtkI0EstimationProjectionFilter_hxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <algorithm>
#include <fstream>

namespace rtk
{
template< class TInputImage, class TOutputImage, unsigned char bitShift >
I0EstimationProjectionFilter< TInputImage, TOutputImage, bitShift >
::I0EstimationProjectionFilter()
{
  m_MaxPixelValue = (InputImagePixelType)((1<<24)-1);
  m_ExpectedI0 = m_MaxPixelValue;
  m_SaveHistograms = false;
  m_Reset = false;

  m_I0 = 0;
  m_I0fwhm = 0;
  m_Np = 0;
  m_I0rls = 0.;
  m_Lambda = 0.8;
  m_DynThreshold = 20;

  m_Imin = 0;
  m_Imax = m_MaxPixelValue;

  m_Mutex = itk::MutexLock::New();
}

template< class TInputImage, class TOutputImage, unsigned char bitShift >
void I0EstimationProjectionFilter< TInputImage, TOutputImage, bitShift >
::BeforeThreadedGenerateData()
{
  m_NBins = (std::vector<unsigned int>::size_type)( (m_MaxPixelValue+1) >>bitShift);
  m_Imax = m_MaxPixelValue;
  m_Histogram.resize(m_NBins, 0);

  std::vector< unsigned int >::iterator it = m_Histogram.begin();

  for (; it != m_Histogram.end(); ++it )
    {
    *it = 0;
    }

  m_Nthreads = this->GetNumberOfThreads();
  m_Nsync = 0;

  if ( m_Reset )
    {
    m_Np = 0;
    }
}

template< class TInputImage, class TOutputImage, unsigned char bitShift >
void I0EstimationProjectionFilter< TInputImage, TOutputImage, bitShift >
::ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                       ThreadIdType itkNotUsed(threadId))
{
  itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<InputImageType>      itOut(this->GetOutput(), outputRegionForThread);

  itIn.GoToBegin();
  itOut.GoToBegin();
  if (this->GetInput() != this->GetOutput()) // If not in place, copy is required
    {
    while (!itIn.IsAtEnd())
      {
      itOut.Set(itIn.Get());
      ++itIn;
      ++itOut;
      }
    }

  // Computation of region histogram

  std::vector< unsigned int > m_thHisto;   // Per-thread histogram
  m_thHisto.resize(m_NBins, 0);

  // Fill in its own histogram
  itIn.GoToBegin();
  while ( !itIn.IsAtEnd() )
    {
    m_thHisto[itIn.Get() >> bitShift]++;
    ++itIn;
    }

  // Merge into global histogram

  m_Mutex->Lock();
    {
    // Add its results to shared histogram
    for ( unsigned int i = 0; i < m_NBins; ++i )
      {
      m_Histogram[i] += m_thHisto[i];
      }

    // The last thread has to do something more
    ++m_Nsync;
    if ( m_Nsync >= m_Nthreads )
      {
      // RMQ 1 : there might be pixels outside the min-max region. They are
      // supposed to be inconsistents (unused detector lines, dead pixels,...)

      // Search for upper bound of the histogram : gives the highest intensity
      // value
      m_Imax = m_NBins - 1;
      while ( ( m_Histogram[m_Imax] <= m_DynThreshold ) && ( m_Imax > 0 ) )
        {
        --m_Imax;
        }
      while ( ( m_Histogram[m_Imax] == 0 ) && ( m_Imax < m_NBins ) ) // Get back
                                                                     // to zero
        {
        ++m_Imax;
        }

      // Search for lower bound of the histogram: gives the lowest intensity
      // value
      m_Imin = 0;
      while ( ( m_Histogram[m_Imin] <= m_DynThreshold ) && ( m_Imin < m_Imax ) )
        {
        ++m_Imin;
        }
      while ( ( m_Histogram[m_Imin] == 0 ) && ( m_Imin > 0 ) ) // Get back to
                                                               // zero
        {
        --m_Imin;
        }

      m_Imin = ( m_Imin << bitShift );
      m_Imax = ( m_Imax << bitShift );

      // If Imax near zero - Potentially no exposure
      // If Imin near Imax - problem to be fixed - No object
      // If Imax very close to MaxPixelValue then possible saturation
      }
    }
  m_Mutex->Unlock();
}

template< class TInputImage, class TOutputImage, unsigned char bitShift >
void I0EstimationProjectionFilter< TInputImage, TOutputImage, bitShift >
::AfterThreadedGenerateData()
{
  // Search for the background mode in the last quarter of the histogram

  unsigned int        startIdx = (3 * (m_Imax >> 2)) >> bitShift;
  unsigned int        idx = startIdx;
  unsigned int        maxId = startIdx;
  InputImagePixelType maxVal = m_Histogram[startIdx];

  while (idx < (unsigned int)(m_Imax >> bitShift))
  {
    if (m_Histogram[idx] >= maxVal)
    {
      maxVal = m_Histogram[idx];
      maxId = idx;
    }
    ++idx;
  }
  m_I0 = InputImagePixelType((maxId) << bitShift);
  m_I0rls = (m_Np > 1) ? (InputImagePixelType)((float)(m_I0rls * m_Lambda) + (float)(m_I0)* (1. - m_Lambda)) : m_I0;

  // If estimated I0 at the boundaries, either startIdx or Imax then we missed
  // smth or no background mode

  InputImagePixelType widthval = (InputImagePixelType) (float) (maxVal >> 1);
  unsigned int lowBound = maxId;
  while ( ( m_Histogram[lowBound] > widthval ) && ( lowBound > 0 ) )
    {
    lowBound--;
    }

  unsigned int highBound = maxId;
  while ( ( m_Histogram[highBound] > widthval ) && ( highBound < m_Imax ) )
    {
    highBound++;
    }

  unsigned int peakFwhm = ( ( highBound - lowBound ) << bitShift );
  m_I0fwhm = peakFwhm;

  m_LowBound = ( lowBound << bitShift );
  m_HighBound = ( highBound << bitShift );

  ++m_Np;

  if ( m_SaveHistograms )
    {
    std::ofstream paramFile;
    paramFile.open("i0est_histogram.csv", std::ofstream::out | std::ofstream::app);
    std::vector< unsigned int >::const_iterator itf = m_Histogram.begin();
    for (; itf != m_Histogram.end(); ++itf )
      {
      paramFile << *itf << ",";
      }
    paramFile.close();
    }
}
} // end namespace rtk
#endif
