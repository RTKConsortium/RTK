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

#ifndef rtkSignalToInterpolationWeights_h
#define rtkSignalToInterpolationWeights_h

#include "itkCSVFileReaderBase.h"
#include "itkArray2D.h"

namespace rtk
{

/** \class SignalToInterpolationWeights
 * \brief Computes interpolation weights for 4D reconstruction
 *
 * Computes the interpolation weights (along time) for 4D reconstruction
 * from the input signal (the phase at which each projection has been acquired).
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */

class ITK_EXPORT SignalToInterpolationWeights : public itk::CSVFileReaderBase
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(SignalToInterpolationWeights);
#else
  ITK_DISALLOW_COPY_AND_MOVE(SignalToInterpolationWeights);
#endif

  /** Standard class type alias */
  using Self = SignalToInterpolationWeights;
  using Superclass = CSVFileReaderBase;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro(SignalToInterpolationWeights, itk::CSVFileReaderBase);

  /** DataFrame Object types */
  using Array2DType = itk::Array2D<float>;

  /** The value type of the dataset. */
  using ValueType = float;

  /** Required, but not used */
  void
  Parse() override
  {}

  /** Does the real work */
  virtual void
  Update();

  /** Aliased to the GetDataFrameObject() method to be consistent with the
   *  rest of the pipeline */
  virtual Array2DType
  GetOutput();

  /** Configure the filter to use uneven temporal spacing (finer temporal resolution during systole) */
  itkSetMacro(NumberOfReconstructedFrames, int);
  itkGetMacro(NumberOfReconstructedFrames, int);

  /** Set the input signal */
  void
  SetSignal(const std::vector<double> signal);

protected:
  SignalToInterpolationWeights();
  ~SignalToInterpolationWeights() override = default;

  /** Print the reader. */
  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

private:
  Array2DType         m_Array2D;
  int                 m_NumberOfReconstructedFrames;
  std::vector<double> m_Signal;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSignalToInterpolationWeights.hxx"
#endif

#endif
