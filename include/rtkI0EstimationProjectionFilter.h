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

#ifndef rtkI0EstimationProjectionFilter_h
#define rtkI0EstimationProjectionFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

#include <mutex>
#include <vector>
#include <string>

namespace rtk
{
/** \class I0EstimationProjectionFilter
 *
 * \brief Estimate the I0 value from the projection histograms
 *
 * \author Sebastien Brousmiche
 *
 * \test rtkI0estimationtest.cxx
 *
 * \ingroup RTK InPlaceImageFilter
 */

template <class TInputImage = itk::Image<unsigned short, 3>,
          class TOutputImage = TInputImage,
          unsigned char bitShift = 2>
class ITK_EXPORT I0EstimationProjectionFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(I0EstimationProjectionFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(I0EstimationProjectionFilter);
#endif

  /** Standard class type alias. */
  using Self = I0EstimationProjectionFilter<TInputImage, TOutputImage, bitShift>;
  using Superclass = itk::InPlaceImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(I0EstimationProjectionFilter, ImageToImageFilter);

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using ImagePointer = typename InputImageType::Pointer;
  using ImageConstPointer = typename InputImageType::ConstPointer;
  using OutputImageRegionType = typename Superclass::OutputImageRegionType;
  using InputImagePixelType = typename InputImageType::PixelType;

  itkConceptMacro(InputImagePixelTypeIsInteger, (itk::Concept::IsInteger<InputImagePixelType>));

  /** Main Output: estimation result. */
  itkGetMacro(I0, InputImagePixelType);
  itkGetMacro(I0fwhm, InputImagePixelType);
  itkGetMacro(I0rls, InputImagePixelType);

  /** Maximum encodable detector value if different from (2^16-1). The default
   * is the minimum between 2^24-1 and the numerical limit of the input pixel
   * type. This allows to limit histogram size to 2^(24-bitShift)-1. */
  itkSetMacro(MaxPixelValue, InputImagePixelType);
  itkGetMacro(MaxPixelValue, InputImagePixelType);

  /** Expected I0 value (as a result of a detector calibration) */
  itkSetMacro(ExpectedI0, InputImagePixelType);
  itkGetMacro(ExpectedI0, InputImagePixelType);

  /** RSL estimate coefficient */
  itkSetMacro(Lambda, float);
  itkGetMacro(Lambda, float);

  /** Write Histograms in a csv file
   * Is false by default */
  itkSetMacro(Reset, bool);
  itkGetConstMacro(Reset, bool);
  itkBooleanMacro(Reset);

  /** Write Histograms in a csv file
   * Is false by default */
  itkSetMacro(SaveHistograms, bool);
  itkGetConstMacro(SaveHistograms, bool);
  itkBooleanMacro(SaveHistograms);

protected:
  I0EstimationProjectionFilter();
  ~I0EstimationProjectionFilter() override = default;

  void
  BeforeThreadedGenerateData() override;

  void
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId) override;

  void
  AfterThreadedGenerateData() override;

private:
  // Input variables
  InputImagePixelType m_ExpectedI0;    // Expected I0 value (as a result of a
                                       // detector calibration)
  InputImagePixelType m_MaxPixelValue; // Maximum encodable detector value if
                                       // different from (2^16-1)
  float m_Lambda;                      // RLS coefficient
  bool  m_SaveHistograms;              // Save histograms in a output file
  bool  m_Reset;                       // Reset counters

  // Secondary inputs
  std::vector<unsigned int>::size_type m_NBins; // Histogram size, computed
                                                // from m_MaxPixelValue and bitshift

  // Main variables
  std::vector<unsigned int> m_Histogram; // compressed (bitshifted) histogram
  InputImagePixelType       m_I0;        // I0 estimate with no a priori for
                                         // each new image
  InputImagePixelType m_I0rls;           // Updated RLS estimate
  InputImagePixelType m_I0fwhm;          // FWHM of the I0 mode

  // Secondary variables
  unsigned int m_Np;                    // Number of previously analyzed
                                        // images
  InputImagePixelType m_Imin, m_Imax;   // Define the range of consistent
                                        // pixels in histogram
  unsigned int m_DynThreshold;          // Detector values with a frequency of
                                        // less than dynThreshold outside
                                        // min/max are discarded
  unsigned int m_LowBound, m_HighBound; // Lower/Upper bounds of the I0 mode
                                        // at half width

  std::mutex m_Mutex;
  int        m_Nsync;
  int        m_Nthreads;
};
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkI0EstimationProjectionFilter.hxx"
#endif

#endif
