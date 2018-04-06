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
#ifndef rtkPhaseReader_h
#define rtkPhaseReader_h

#include <itkCSVFileReaderBase.h>

namespace rtk
{

/** \class PhaseReader
 * \brief Parses csv file containing the cardiac or respiratory phases of each projection.
 */

class ITK_EXPORT PhaseReader:public itk::CSVFileReaderBase
{
public:
    /** Standard class typedefs */
    typedef PhaseReader                     Self;
    typedef CSVFileReaderBase               Superclass;
    typedef itk::SmartPointer<Self>         Pointer;
    typedef itk::SmartPointer<const Self>   ConstPointer;

    /** Standard New method. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods) */
    itkTypeMacro(Self,Superclass)

    /** The value type of the dataset. */
    typedef float ValueType;

    /** Parses the data from the file. Gets the phases of the projections
  * into a vector, then generate an Array2D object containing the interpolation weights  */
    void Parse() ITK_OVERRIDE;

    /** Aliased to the Parse() method to be consistent with the rest of the
   * pipeline. */
    virtual void Update();

    /** Aliased to the GetDataFrameObject() method to be consistent with the
   *  rest of the pipeline */
    virtual std::vector<float> GetOutput();

protected:

    PhaseReader();
    ~PhaseReader () {}

    /** Print the reader. */
    void PrintSelf(std::ostream & os, itk::Indent indent) const ITK_OVERRIDE;

private:

    std::vector<float> m_Phases;

    PhaseReader(const Self &);  //purposely not implemented
    void operator=(const Self &);          //purposely not implemented
};

} //end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkPhaseReader.hxx"
#endif

#endif
