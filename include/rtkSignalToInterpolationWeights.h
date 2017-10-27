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
 */

class ITK_EXPORT SignalToInterpolationWeights:public itk::CSVFileReaderBase
{
public:
    /** Standard class typedefs */
    typedef SignalToInterpolationWeights      Self;
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

    /** Required, but not used */
    void Parse() ITK_OVERRIDE {}

    /** Does the real work */
    virtual void Update();

    /** Aliased to the GetDataFrameObject() method to be consistent with the
   *  rest of the pipeline */
    virtual Array2DType GetOutput();

    /** Configure the filter to use uneven temporal spacing (finer temporal resolution during systole) */
    itkSetMacro(NumberOfReconstructedFrames, int)
    itkGetMacro(NumberOfReconstructedFrames, int)

    /** Set the input signal */
    void SetSignal(const std::vector<double> signal);

protected:
    SignalToInterpolationWeights();
    ~SignalToInterpolationWeights () {}

    /** Print the reader. */
    void PrintSelf(std::ostream & os, itk::Indent indent) const ITK_OVERRIDE;

private:
    Array2DType           m_Array2D;
    int                   m_NumberOfReconstructedFrames;
    std::vector<double>   m_Signal;

    SignalToInterpolationWeights(const Self &);  //purposely not implemented
    void operator=(const Self &);          //purposely not implemented
};

} //end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSignalToInterpolationWeights.hxx"
#endif

#endif
