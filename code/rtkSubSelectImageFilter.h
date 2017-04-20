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
#ifndef rtkSubSelectImageFilter_h
#define rtkSubSelectImageFilter_h

#include <itkPasteImageFilter.h>
#include <itkExtractImageFilter.h>
#include "rtkConstantImageSource.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class SubSelectImageFilter
 * \brief Subselects projections from a stack of projections
 *
 * This abstract class takes as input a stack of projections and the
 * corresponding geometry and creates an output stack of projections and
 * its corresponding geometry using the two members m_NbSelectedProjs and
 * m_SelectedProjections. The members must be set before
 * GenerateOutputInformation is called. Streaming of the output is possible.
 * The output is produced from the following mini-pipeline:
 *
 * \dot
 * digraph SubSelectImageFilter {
 * Input [label="Input (Projections)", shape=Mdiamond];
 * Output [label="Output (Projections)", shape=Mdiamond];
 *
 * node [shape=box];
 *
 * Constant [label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * Extract [label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * Paste [label="itk::PasteImageFilter" URL="\ref itk::PasteImageFilter"];
 *
 * Input->Extract
 * Extract->Paste
 * Paste->Output
 * Constant->Paste
 * }
 * \enddot
 *
 * \test rtkadmmtotalvariationtest.cxx, rtkselectoneprojpercycletest.cxx
 *
 * \author Simon Rit
 */
template< typename ProjectionStackType>
class SubSelectImageFilter : public itk::ImageToImageFilter<ProjectionStackType, ProjectionStackType>
{
public:
  /** Standard class typedefs. */
  typedef SubSelectImageFilter                                              Self;
  typedef itk::ImageToImageFilter<ProjectionStackType, ProjectionStackType> Superclass;
  typedef itk::SmartPointer< Self >                                         Pointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(SubSelectImageFilter, itk::ImageToImageFilter)

  /** The set of projections from which a subset will be extracted */
  void SetInputProjectionStack(const ProjectionStackType* Projections);
  typename ProjectionStackType::ConstPointer GetInputProjectionStack();

  typedef itk::PasteImageFilter<ProjectionStackType>                        PasteFilterType;
  typedef itk::ExtractImageFilter<ProjectionStackType, ProjectionStackType> ExtractFilterType;
  typedef rtk::ConstantImageSource<ProjectionStackType>                     EmptyProjectionStackSourceType;
  typedef rtk::ThreeDCircularProjectionGeometry                             GeometryType;

  itkSetMacro(InputGeometry, GeometryType::Pointer)
  itkGetMacro(InputGeometry, GeometryType::Pointer)

  itkGetMacro(SelectedProjections, std::vector<bool>)

  GeometryType::Pointer GetOutputGeometry();

protected:
  SubSelectImageFilter();
  ~SubSelectImageFilter() {}

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

  /** Does the real work. */
  void GenerateData() ITK_OVERRIDE;

  /** Member variables */
  GeometryType::Pointer     m_InputGeometry;
  GeometryType::Pointer     m_OutputGeometry;
  std::vector<bool>         m_SelectedProjections;
  int                       m_NbSelectedProjs;

private:
  SubSelectImageFilter(const Self &); // purposely not implemented
  void operator=(const Self &);       // purposely not implemented

  typename EmptyProjectionStackSourceType::Pointer m_EmptyProjectionStackSource;
  typename ExtractFilterType::Pointer              m_ExtractFilter;
  typename PasteFilterType::Pointer                m_PasteFilter;
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSubSelectImageFilter.hxx"
#endif

#endif
