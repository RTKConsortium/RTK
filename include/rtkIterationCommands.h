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

#ifndef rtkIterationCommands_h
#define rtkIterationCommands_h

#include <itkImageFileWriter.h>

namespace rtk
{

/** \class IterationCommand
 * \brief Abstract superclass to all iteration callbacks.
 * Derived classes must implement the Run() method. Run() can be triggered
 * only once in every n iterations.
 *
 * \author Aurélien Coussat
 *
 * \ingroup RTK
 *
 */
template <typename TCaller>
class IterationCommand : public itk::Command
{
protected:
  /** How many times this command has been executed. */
  unsigned int m_IterationCount = 0;

  /** Trigger the callback every n iterations. */
  unsigned int m_TriggerEvery = 1;

  /** Callback function to be redefined by derived classes. */
  virtual void
  Run(const TCaller * caller) = 0;

public:
  /** Standard class typedefs. */
  typedef IterationCommand        Self;
  typedef itk::Command            Superclass;
  typedef itk::SmartPointer<Self> Pointer;

  itkSetMacro(TriggerEvery, unsigned int)

    void Execute(itk::Object * caller, const itk::EventObject & event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object * caller, const itk::EventObject & event) override
  {
    if (!itk::IterationEvent().CheckEvent(&event))
    {
      return;
    }
    ++m_IterationCount;
    if ((m_IterationCount % m_TriggerEvery) == 0)
    {
      const auto * cCaller = dynamic_cast<const TCaller *>(caller);
      if (cCaller)
      {
        Run(cCaller);
      } // TODO fail when cast fails
    }
  }
};

/** \class VerboseIterationCommand
 * \brief Outputs to standard output when an iteration completes.
 *
 * \author Aurélien Coussat
 *
 * \ingroup RTK
 *
 */
template <typename TCaller>
class VerboseIterationCommand : public IterationCommand<TCaller>
{
public:
  /** Standard class typedefs. */
  typedef VerboseIterationCommand   Self;
  typedef IterationCommand<TCaller> Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro(Self);

protected:
  void
  Run(const TCaller * itkNotUsed(caller)) override
  {
    std::cout << "Iteration " << this->m_IterationCount << " completed."
              << std::endl; // TODO allow string personnalization
  }
};

/** \class OutputIterationCommand
 * \brief Output intermediate iterations in a file.
 * This class is useful to check convergence of an iterative method
 * and to avoid starting over similar computations when testing
 * hyperparameters of an iterative algorithm.
 *
 * \author Aurélien Coussat
 *
 * \ingroup RTK
 *
 */
template <typename TCaller, typename TOutputImage>
class OutputIterationCommand : public IterationCommand<TCaller>
{
public:
  /** Standard class typedefs. */
  typedef OutputIterationCommand    Self;
  typedef IterationCommand<TCaller> Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro(Self);

  itkSetMacro(FileFormat, std::string)

    protected
    :

    /** Output file name, where %d is the current iteration number */
    std::string m_FileFormat;

  void
  Run(const TCaller * caller) override
  {
    typedef itk::ImageFileWriter<TOutputImage> WriterType;
    typename WriterType::Pointer               writer = WriterType::New();

    char         buffer[100];
    unsigned int size = sprintf(buffer, m_FileFormat.c_str(), this->m_IterationCount);
    writer->SetFileName(std::string(buffer, size));

    writer->SetInput(caller->GetOutput());
    writer->Update();
  }
};

} // end namespace rtk

#endif
