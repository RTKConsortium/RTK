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

#ifndef rtkPhasesToInterpolationWeights_h
#define rtkPhasesToInterpolationWeights_h

#include "itkCSVFileReaderBase.h"
#include "itkArray2D.h"

namespace rtk
{

/** \class PhasesToInterpolationWeights
 * \brief Parses csv file containing the cardiac or respiratory phases of each projection, and generates interpolation weights for 4D reconstruction.
 *
 * Useful for 4D reconstruction of beating heart or breathing thorax
 *
 */

class ITK_EXPORT PhasesToInterpolationWeights:public itk::CSVFileReaderBase
{
public:
    /** Standard class typedefs */
    typedef PhasesToInterpolationWeights      Self;
    typedef CSVFileReaderBase                 Superclass;
    typedef itk::SmartPointer<Self>           Pointer;
    typedef itk::SmartPointer<const Self>     ConstPointer;

    /** Standard New method. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods) */
    itkTypeMacro(Self,Superclass)

    /** DataFrame Object types */
    typedef itk::Array2D<float>    Array2DType;

    /** The value type of the dataset. */
    typedef float ValueType;

    //  /** This method can be used to get the data frame object once the data from
    //  * the file has been parsed. */
    //  itkGetObjectMacro(Array2D,Array2DType);

    /** Parses the data from the file. Gets the phases of the projections
  * into a vector, then generate an Array2D object containing the interpolation weights  */
    void Parse() ITK_OVERRIDE;

    /** Aliased to the Parse() method to be consistent with the rest of the
   * pipeline. */
    virtual void Update();

    /** Aliased to the GetDataFrameObject() method to be consistent with the
   *  rest of the pipeline */
    virtual Array2DType GetOutput();

    /** Configure the filter to use uneven temporal spacing (finer temporal resolution during systole) */
    itkSetMacro(UnevenTemporalSpacing, bool)
    itkGetMacro(UnevenTemporalSpacing, bool)

    /** Configure the filter to use uneven temporal spacing (finer temporal resolution during systole) */
    itkSetMacro(NumberOfReconstructedFrames, int)
    itkGetMacro(NumberOfReconstructedFrames, int)

    /** Set/Get for a list of booleans indicating whether or not each projection must be selected */
    void SetSelectedProjections(std::vector<bool> sprojs);
    itkGetMacro(SelectedProjections, std::vector<bool>)

protected:

    PhasesToInterpolationWeights();
    ~PhasesToInterpolationWeights () {}

    /** Print the reader. */
    void PrintSelf(std::ostream & os, itk::Indent indent) const ITK_OVERRIDE;

private:

    Array2DType         m_Array2D;
    int                 m_NumberOfReconstructedFrames;
    bool                m_UnevenTemporalSpacing;
    std::vector<bool>   m_SelectedProjections;

    PhasesToInterpolationWeights(const Self &);  //purposely not implemented
    void operator=(const Self &);          //purposely not implemented
};

} //end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkPhasesToInterpolationWeights.hxx"
#endif

#endif
