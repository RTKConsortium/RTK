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

#ifndef rtkPolynomialGainCorrectionImageFilter_h
#define rtkPolynomialGainCorrectionImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkSimpleFastMutexLock.h>

#include <vector>

#include "rtkMacro.h"

/** \class PolynomialGainCorrection
 * \brief Pixel-wize polynomial gain calibration
 *
 * Based on 'An improved method for flat-field correction of flat panel x-ray detector'
 *          Kwan, Med. Phys 33 (2), 2006
 * Only allow unsigned short as input format
 *
 * \author Sebastien Brousmiche
 *
 * \ingroup ImageToImageFilter
 */

namespace rtk
{

template<class TInputImage, class TOutputImage>
class PolynomialGainCorrectionImageFilter :
public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef PolynomialGainCorrectionImageFilter                Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                          InputImageType;
  typedef TOutputImage                         OutputImageType;
  typedef typename InputImageType::Pointer     InputImagePointer;
  typedef typename OutputImageType::Pointer    OutputImagePointer;
  typedef typename InputImageType::RegionType  InputImageRegionType;
  typedef typename TOutputImage::RegionType    OutputImageRegionType;
  typedef typename std::vector< float >        VectorType;
  typedef typename OutputImageType::SizeType   OutputSizeType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(PolynomialGainCorrectionImageFilter, itk::ImageToImageFilter);

  /** Dark image, 2D same size of one input projection */
  void SetDarkImage(const InputImagePointer gain);

  /** Weights, matrix A from reference paper
   *  3D image: 2D x order. */
  void SetGainCoefficients(const OutputImagePointer gain);

  /* if K==0, the filter is bypassed */
  itkSetMacro(K, float);
  itkGetMacro(K, float);

protected:
  PolynomialGainCorrectionImageFilter();
  ~PolynomialGainCorrectionImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType threadId ) ITK_OVERRIDE;

private:
  //purposely not implemented
  PolynomialGainCorrectionImageFilter(const Self&);
  void operator=(const Self&);

protected:
  bool               m_MapsLoaded;        // True if gain maps loaded
  int                m_ModelOrder;        // Polynomial correction order
  float              m_K;                 // Scaling constant, a 0 means no correction
  VectorType         m_PowerLut;          // Vector containing I^n
  InputImagePointer  m_DarkImage;         // Dark image
  OutputImagePointer m_GainImage;         // Gain coefficients (A matrix)
  OutputSizeType     m_GainSize;          // Gain map size
}; // end of class

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkPolynomialGainCorrectionImageFilter.hxx"
#endif

#endif
