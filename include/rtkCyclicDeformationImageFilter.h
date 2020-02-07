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

#ifndef rtkCyclicDeformationImageFilter_h
#define rtkCyclicDeformationImageFilter_h

#include <itkImageToImageFilter.h>

#include "rtkConfiguration.h"
#include "rtkMacro.h"

namespace rtk
{

/** \class CyclicDeformationImageFilter
 * \brief Return 3D deformation vector field according to input 4D vector field,
 * phase signal and frame number.
 *
 * The 4D deformation vector field (DVF) describes the deformation along one cycle.
 * The phase signal is passed via a file name pointing to a text file with one
 * value per line. It must be in the inteval [0,1), 0 meaning frame 0 of the 4D
 * DVF, 0.5 the middle frame (if the number of frames of the 4D DVF is odd), etc.
 * The frame number is the value of the signal for which we wish to obtain the
 * resulting 3D DVF. Linear interpolation is used to compute that DVF.
 * This cyclic deformation model has been described in [Rit et al, TMI, 2009] and
 * [Rit et al, Med Phys, 2009].
 *
 * \test rtkmotioncompensatedfdktest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT CyclicDeformationImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(CyclicDeformationImageFilter);

  /** Standard class type alias. */
  using Self = CyclicDeformationImageFilter;
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using Superclass = itk::ImageToImageFilter<InputImageType, OutputImageType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using OutputImageRegionType = typename OutputImageType::RegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CyclicDeformationImageFilter, itk::ImageToImageFilter);

  /** Get / Set the signal file name relating each frame to a phase position.
      The signal file is a text file containing one line per frame. */
  itkGetMacro(SignalFilename, std::string);
  virtual void
  SetSignalFilename(const std::string _arg);
  virtual void
  SetSignalVector(std::vector<double> _arg);

  /** Get / Set the frame number. The number is used to lookup in the signal file
   * which phase value should be used to interpolate. */
  itkGetMacro(Frame, unsigned int);
  itkSetMacro(Frame, unsigned int);

protected:
  CyclicDeformationImageFilter() {}
  ~CyclicDeformationImageFilter() override = default;

  void
  GenerateOutputInformation() override;
  void
  GenerateInputRequestedRegion() override;
  void
  BeforeThreadedGenerateData() override;
  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

  // Linear interpolation position and weights
  unsigned int m_FrameInf;
  unsigned int m_FrameSup;
  double       m_WeightInf;
  double       m_WeightSup;

private:
  unsigned int m_Frame{ 0 };

  std::string         m_SignalFilename;
  std::vector<double> m_Signal;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkCyclicDeformationImageFilter.hxx"
#endif

#endif
