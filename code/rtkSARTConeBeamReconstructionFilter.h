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

#ifndef __rtkSARTConeBeamReconstructionFilter_h
#define __rtkSARTConeBeamReconstructionFilter_h

#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

#include <itkExtractImageFilter.h>
#if ITK_VERSION_MAJOR <= 3
#  include <itkMultiplyByConstantImageFilter.h>
#else
#  include <itkMultiplyImageFilter.h>
#endif
#include <itkSubtractImageFilter.h>
#include <itkTimeProbe.h>

namespace rtk
{

/** \class SARTConeBeamReconstructionFilter
 * \brief Implements the Simultaneous Algebraic Reconstruction Technique [Andersen, 1984]
 *
 * SARTConeBeamReconstructionFilter is a mini-pipeline filter which combines
 * the different steps of the SART cone-beam reconstruction, mainly:
 * - ExtractFilterType to work on one projection at a time
 * - ForwardProjectionImageFilter,
 * - SubtractImageFilter,
 * - BackProjectionImageFilter.
 * The input stack of projections is processed piece by piece (the size is
 * controlled with ProjectionSubsetSize) via the use of itk::ExtractImageFilter
 * to extract sub-stacks.
 *
 * \author Simon Rit
 *
 * \ingroup ReconstructionAlgorithm
 */
template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT SARTConeBeamReconstructionFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SARTConeBeamReconstructionFilter                   Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Typedefs of each subfilter of this composite filter */
  typedef itk::ExtractImageFilter< InputImageType, OutputImageType >                     ExtractFilterType;
#if ITK_VERSION_MAJOR <= 3
  typedef itk::MultiplyByConstantImageFilter< OutputImageType, double, OutputImageType > MultiplyFilterType;
#else
  typedef itk::MultiplyImageFilter< OutputImageType, OutputImageType, OutputImageType >  MultiplyFilterType;
#endif
  typedef rtk::ForwardProjectionImageFilter< OutputImageType, OutputImageType >          ForwardProjectionFilterType;
  typedef itk::SubtractImageFilter< OutputImageType, OutputImageType >                   SubtractFilterType;
  typedef rtk::BackProjectionImageFilter< OutputImageType, OutputImageType >             BackProjectionFilterType;
  typedef typename BackProjectionFilterType::Pointer                                     BackProjectionFilterPointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(SARTConeBeamReconstructionFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);

  void PrintTiming(std::ostream& os) const;

  /** Get / Set the number of iterations. Default is 3. */
  itkGetMacro(NumberOfIterations, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);

  /** Get / Set the convergence factor. Default is 0.3. */
  itkGetMacro(Lambda, double);
  itkSetMacro(Lambda, double);

  /** Set and init the backprojection filter. Default is voxel based backprojection. */
  virtual void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

protected:
  SARTConeBeamReconstructionFilter();
  ~SARTConeBeamReconstructionFilter(){}

  virtual void GenerateInputRequestedRegion();

  virtual void GenerateOutputInformation();

  virtual void GenerateData();

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

  /** Pointers to each subfilter of this composite filter */
  typename ExtractFilterType::Pointer           m_ExtractFilter;
  typename MultiplyFilterType::Pointer          m_ZeroMultiplyFilter;
  typename ForwardProjectionFilterType::Pointer m_ForwardProjectionFilter;
  typename SubtractFilterType::Pointer          m_SubtractFilter;
  typename MultiplyFilterType::Pointer          m_MultiplyFilter;
  typename BackProjectionFilterType::Pointer    m_BackProjectionFilter;

private:
  //purposely not implemented
  SARTConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

  /** Geometry object */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Number of projections processed at a time. */
  unsigned int m_NumberOfIterations;

  /** Convergence factor according to Andersen's publications which relates
   * to the step size of the gradient descent. Default 0.3, Must be in (0,2). */
  double m_Lambda;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSARTConeBeamReconstructionFilter.txx"
#endif

#endif
