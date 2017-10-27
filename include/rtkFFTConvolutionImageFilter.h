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

#ifndef rtkFFTConvolutionImageFilter_h
#define rtkFFTConvolutionImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkConceptChecking.h>

#include "rtkConfiguration.h"
#include "rtkMacro.h"

namespace rtk
{

/** \class FFTConvolutionImageFilter
 * \brief Base class for 1D or 2D FFT based convolution of projections
 *
 * The filter code is based on FFTConvolutionImageFilter by Gaetan Lehmann
 * (see http://hdl.handle.net/10380/3154).
 *
 * \test rtkrampfiltertest.cxx, rtkscatterglaretest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */

template<class TInputImage, class TOutputImage, class TFFTPrecision>
class ITK_EXPORT FFTConvolutionImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FFTConvolutionImageFilter                          Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                       InputImageType;
  typedef TOutputImage                                      OutputImageType;
  typedef typename InputImageType::RegionType               RegionType;
  typedef typename InputImageType::IndexType                IndexType;
  typedef typename InputImageType::SizeType                 SizeType;

  typedef typename itk::Image<TFFTPrecision,
                              TInputImage::ImageDimension > FFTInputImageType;
  typedef typename FFTInputImageType::Pointer               FFTInputImagePointer;
  typedef typename itk::Image<std::complex<TFFTPrecision>,
                              TInputImage::ImageDimension > FFTOutputImageType;
  typedef typename FFTOutputImageType::Pointer              FFTOutputImagePointer;
  typedef itk::Vector<int,2>                                ZeroPadFactorsType;

  /** ImageDimension constants */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Runtime information support. */
  itkTypeMacro(FFTConvolutionImageFilter, ImageToImageFilter);

  /**
   * Set/Get the greatest prime factor allowed on the size of the padded image.
   * The filter increase the size of the image to reach a size with the greatest
   * prime factor smaller or equal to the specified value. The default value is
   * 13, which is the greatest prime number for which the FFT are precomputed
   * in FFTW, and thus gives very good performance.
   * A greatest prime factor of 2 produce a size which is a power of 2, and thus
   * is suitable for vnl base fft filters.
   * A greatest prime factor of 1 or less - typically 0 - disable the extra padding.
   *
   * Warning: this parameter is not used (and useful) only when ITK is built with
   * FFTW support.
   */
  itkGetConstMacro(GreatestPrimeFactor, int);
  itkSetMacro(GreatestPrimeFactor, int);

  /** Set/Get the percentage of the image widthfeathered with data to correct
    * for truncation.
    */
  itkGetConstMacro(TruncationCorrection, double);
  itkSetMacro(TruncationCorrection, double);

  /** Set/Get the zero padding factors in x and y directions. Accepted values
    * are either 1 and 2. The y value is only used if the convolution kernel is 2D.
    */
  itkGetConstMacro(ZeroPadFactors, ZeroPadFactorsType);
  virtual void SetZeroPadFactors (ZeroPadFactorsType _arg)
    {
    if (m_ZeroPadFactors != _arg)
      {
      m_ZeroPadFactors = _arg;
      m_ZeroPadFactors[0] = std::max(m_ZeroPadFactors[0], 1);
      m_ZeroPadFactors[1] = std::max(m_ZeroPadFactors[1], 1);
      m_ZeroPadFactors[0] = std::min(m_ZeroPadFactors[0], 2);
      m_ZeroPadFactors[1] = std::min(m_ZeroPadFactors[1], 2);
      this->Modified();
      }
    }

protected:
  FFTConvolutionImageFilter();
  ~FFTConvolutionImageFilter() {}

  virtual void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  void AfterThreadedGenerateData() ITK_OVERRIDE;

  void ThreadedGenerateData( const RegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

  /** Pad the inputRegion region of the input image and returns a pointer to the new padded image.
    * Padding includes a correction for truncation [Ohnesorge, Med Phys, 2000].
    * centralRegion is the region of the returned image which corresponds to inputRegion.
    */
  virtual FFTInputImagePointer PadInputImageRegion(const RegionType &inputRegion);
  RegionType GetPaddedImageRegion(const RegionType &inputRegion);

  void PrintSelf(std::ostream& os, itk::Indent indent) const ITK_OVERRIDE;

  bool IsPrime( int n ) const;

  int GreatestPrimeFactor( int n ) const;

  /** Creates and return a pointer to the convolution kernel. Can be 1D or 2D.
   *  Used in generate data functions, must be implemented in daughter classes.  */
  virtual void UpdateFFTConvolutionKernel(const SizeType size) = 0;

  /** Pre compute weights for truncation correction in a lookup table. The index
    * is the distance to the original image border.
    * Careful: the function is not thread safe but it does nothing if the weights have
    * already been computed.
    */
  virtual void UpdateTruncationMirrorWeights();
  typename std::vector<TFFTPrecision> m_TruncationMirrorWeights;

    /** Must be set to fix whether the kernel is 1D or 2D. Will have an effect on
   * the padded region and the input requested region. */
  int m_KernelDimension;

  /**
    * FFT of the convolution kernel that each daughter class must update.
    */
  FFTOutputImagePointer m_KernelFFT;

private:
  FFTConvolutionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** Percentage of the image width which is feathered with data to correct for truncation.
    * 0 (default) means no correction.
    */
  double m_TruncationCorrection;
  int GetTruncationCorrectionExtent();

  /** Zero padding factors in x and y directions. Accepted values are either 1
    * and 2. The y value is only used if the convolution kernel is 2D.
    */
  ZeroPadFactorsType m_ZeroPadFactors;

  /**
   * Greatest prime factor of the FFT input.
   */
  int m_GreatestPrimeFactor;
  int m_BackupNumberOfThreads;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFFTConvolutionImageFilter.hxx"
#endif

#endif
