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

#ifndef rtkExtractPhaseImageFilter_h
#define rtkExtractPhaseImageFilter_h

#include <itkInPlaceImageFilter.h>

namespace rtk
{
  /** \class ExtractPhaseImageFilter
   *
   * \brief Extracts the phase of a 1D signal
   *
   * The filter extracts the phase of a 1D periodic signal, which is useful in
   * gated CT. There are several ways of defining the phase of a signal and the
   * filter offers two of them:
   * - the phase of the complex analytic signal as defined in
   * rtk::HilbertImageFilter,
   * - linear between chosen extrema, either minima or maxima.
   * There are several parameters to the extraction that should be at least
   * adjusted for every system and possibly for some patients.
   *
   * \test rtkamsterdamshroudtest.cxx
   *
   * \author Simon Rit
   *
   * \ingroup ImageToImageFilter
   */

template<class TImage>
class ITK_EXPORT ExtractPhaseImageFilter :
  public itk::InPlaceImageFilter<TImage>
{
public:
  /** Standard class typedefs. */
  typedef ExtractPhaseImageFilter         Self;
  typedef itk::InPlaceImageFilter<TImage> Superclass;
  typedef itk::SmartPointer<Self>         Pointer;
  typedef itk::SmartPointer<const Self>   ConstPointer;

  /** Convenient typedefs. */
  typedef typename TImage::SizeType::SizeValueType KernelSizeType;
  typedef std::vector<int>                         PositionsListType;
  typedef enum {LOCAL_PHASE=0,
                LINEAR_BETWEEN_MINIMA,
                LINEAR_BETWEEN_MAXIMA}             ModelType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ExtractPhaseImageFilter, itk::InPlaceImageFilter);

  /** The input signal may be smoothed before taking the phase of the Hilbert
   * transform. This parameter sets the number of samples for this smoothing. */
  itkSetMacro(MovingAverageSize, KernelSizeType)
  itkGetMacro(MovingAverageSize, KernelSizeType)

  /** Low frequencies of the signal are removed before taking the phase of the
   * Hilbert transform using an unsharp mask, i.e., the difference of the
   * signal and its moving average. This parameter sets the number of samples
   * used for the moving average, default is 55. */
  itkSetMacro(UnsharpMaskSize, KernelSizeType)
  itkGetMacro(UnsharpMaskSize, KernelSizeType)

  /** During the update, extrema are extracted and can be retrieved after an
   * update of the output. */
  itkGetMacro(MinimaPositions, PositionsListType)
  itkGetMacro(MaximaPositions, PositionsListType)

  /** After smoothing and unsharping, you can chose the model for the phase
   * extraction which describes the position in the respiratory cycle by a
   * number between 0 and 1:
   * - LOCAL_PHASE: local phase, i.e., phase of the Hilbert transform of the signal,
   * - LINEAR_BETWEEN_MINIMA (LINEAR_BETWEEN_MAXIMA): phase phase is linear
   * between two consecutive minima (maxima), with the minima (maxima) at 0.
   * Default is LINEAR_BETWEEN_MINIMA. */
  itkSetMacro(Model, ModelType)
  itkGetMacro(Model, ModelType)

protected:
  ExtractPhaseImageFilter();
  ~ExtractPhaseImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

private:
  ExtractPhaseImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);          //purposely not implemented

  void ComputeLinearPhaseBetweenPositions(const PositionsListType & positions);

  KernelSizeType    m_MovingAverageSize;
  KernelSizeType    m_UnsharpMaskSize;
  PositionsListType m_MinimaPositions;
  PositionsListType m_MaximaPositions;
  ModelType         m_Model;
}; // end of class

} // end of namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkExtractPhaseImageFilter.hxx"
#endif

#endif
