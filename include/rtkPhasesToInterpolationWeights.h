/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkPhasesToInterpolationWeights_h
#define rtkPhasesToInterpolationWeights_h

#include "RTKExport.h"
#include "itkCSVFileReaderBase.h"
#include "itkArray2D.h"

namespace rtk
{

/** \class PhasesToInterpolationWeights
 * \brief Parses csv file containing the cardiac or respiratory phases of each projection, and generates interpolation
 * weights for 4D reconstruction.
 *
 * Useful for 4D reconstruction of beating heart or breathing thorax
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */

class RTK_EXPORT PhasesToInterpolationWeights : public itk::CSVFileReaderBase
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(PhasesToInterpolationWeights);

  /** Standard class type alias */
  using Self = PhasesToInterpolationWeights;
  using Superclass = CSVFileReaderBase;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkOverrideGetNameOfClassMacro(PhasesToInterpolationWeights);

  /** DataFrame Object types */
  using Array2DType = itk::Array2D<float>;

  /** The value type of the dataset. */
  using ValueType = float;

  //  /** This method can be used to get the data frame object once the data from
  //  * the file has been parsed. */
  //  itkGetModifiableObjectMacro(Array2D,Array2DType);

  /** Parses the data from the file. Gets the phases of the projections
   * into a vector, then generate an Array2D object containing the interpolation weights  */
  void
  Parse() override;

  /** Aliased to the Parse() method to be consistent with the rest of the
   * pipeline. */
  virtual void
  Update();

  /** Aliased to the GetDataFrameObject() method to be consistent with the
   *  rest of the pipeline */
  virtual Array2DType
  GetOutput();

  /** Configure the filter to use uneven temporal spacing (finer temporal resolution during systole) */
  itkSetMacro(UnevenTemporalSpacing, bool);
  itkGetMacro(UnevenTemporalSpacing, bool);

  /** Configure the filter to use uneven temporal spacing (finer temporal resolution during systole) */
  itkSetMacro(NumberOfReconstructedFrames, int);
  itkGetMacro(NumberOfReconstructedFrames, int);

  /** Set/Get for a list of booleans indicating whether or not each projection must be selected */
  void
  SetSelectedProjections(std::vector<bool> sprojs);
  itkGetMacro(SelectedProjections, std::vector<bool>);

protected:
  PhasesToInterpolationWeights();
  ~PhasesToInterpolationWeights() override = default;

  /** Print the reader. */
  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

private:
  Array2DType       m_Array2D;
  int               m_NumberOfReconstructedFrames;
  bool              m_UnevenTemporalSpacing;
  std::vector<bool> m_SelectedProjections;
};

} // end namespace rtk

#endif
