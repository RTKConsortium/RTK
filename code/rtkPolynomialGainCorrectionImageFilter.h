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

#ifndef __rtkPolynomialGainCorrectionImageFilter_h
#define __rtkPolynomialGainCorrectionImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkSimpleFastMutexLock.h>

#include <vector>

/** \class PolynomialGainCorrection 
 * \brief Pixel-wize polynomial gain calibration  
 *
 * Based on 'An improved method for flat-field correction of flat panel x-ray detector'
 *          Kwan, Med. Phys 33 (2), 2006
 *
 * \author Sebastien Brousmiche
 *
 * \ingroup ImageToImageFilter
 */

namespace rtk 
{

template<class TInputImage, class TOutputImage=TInputImage>
class PolynomialGainCorrectionImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef PolynomialGainCorrectionImageFilter                           Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                          InputImageType;
  typedef TOutputImage                         OutputImageType;
  typedef typename InputImageType::Pointer     InputImagePtr;
  typedef typename OutputImageType::Pointer    OutputImagePtr;
  typedef typename TInputImage::RegionType     InputImageRegionType;
  typedef typename TOutputImage::RegionType    OutputImageRegionType;
  typedef typename std::vector< double >       VectorType;
  typedef typename OutputImageType::SizeType   OutputSizeType;
  
  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(PolynomialGainCorrectionImageFilter, itk::ImageToImageFilter);

  /** Dark image */
  void SetDarkImage(const InputImagePtr gain);

  /** Weights, matrix A from reference paper */
  void SetGainCoefficients(const OutputImagePtr gain);

  /* if K==0, the filter is bypassed */
  itkSetMacro(K, float);
  itkGetMacro(K, float);

protected:
  PolynomialGainCorrectionImageFilter();
  ~PolynomialGainCorrectionImageFilter() {}

  virtual void GenerateOutputInformation();

  virtual void GenerateInputRequestedRegion();

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType threadId );
  
private:
  //purposely not implemented
  PolynomialGainCorrectionImageFilter(const Self&);
  void operator=(const Self&);

  bool               m_MapsLoaded;
  int                m_ModelOrder;
  float              m_K;                 // Scaling constant
  int                m_NpixValues;        // Also maximum acceptable value in the LUT
  VectorType         m_PowerLut;          // Vector containing I^n
  InputImagePtr      m_DarkImage;         // Dark image
  OutputImagePtr     m_GainImage;         // Gain coefficients (A matrix)
  OutputSizeType     m_GainSize;
}; // end of class

} 

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkPolynomialGainCorrectionImageFilter.hxx"
#endif

#endif

