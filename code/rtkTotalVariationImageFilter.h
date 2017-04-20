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

#ifndef rtkTotalVariationImageFilter_h
#define rtkTotalVariationImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkNumericTraits.h>
#include <itkArray.h>
#include <itkSimpleDataObjectDecorator.h>

#include "rtkMacro.h"

namespace rtk
{
/** \class TotalVariationImageFilter
 * \brief Compute the total variation of an Image.
 *
 * TotalVariationImageFilter computes the total variation, defined
 * as the L1 norm of the image of the L2 norm of the gradient,
 * of an image. The filter needs all of its input image.  It
 * behaves as a filter with an input and output. Thus it can be inserted
 * in a pipeline with other filters and the total variation will only be
 * recomputed if a downstream filter changes.
 *
 * The filter passes its input through unmodified. The filter is
 * threaded.
 *
 * \ingroup MathematicalStatisticsImageFilters
 * \ingroup ITKImageStatistics
 *
 */

template< typename TInputImage >
class TotalVariationImageFilter:
  public itk::ImageToImageFilter< TInputImage, TInputImage >
{
public:
  /** Standard Self typedef */
  typedef TotalVariationImageFilter                           Self;
  typedef itk::ImageToImageFilter< TInputImage, TInputImage > Superclass;
  typedef itk::SmartPointer< Self >                           Pointer;
  typedef itk::SmartPointer< const Self >                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(TotalVariationImageFilter, ImageToImageFilter);

  /** Image related typedefs. */
  typedef typename TInputImage::Pointer InputImagePointer;

  typedef typename TInputImage::RegionType RegionType;
  typedef typename TInputImage::SizeType   SizeType;
  typedef typename TInputImage::IndexType  IndexType;
  typedef typename TInputImage::PixelType  PixelType;

  /** Image related typedefs. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  /** Type to use for computations. */
  typedef typename itk::NumericTraits< PixelType >::RealType RealType;

  /** Smart Pointer type to a DataObject. */
  typedef typename itk::DataObject::Pointer DataObjectPointer;

  /** Type of DataObjects used for scalar outputs */
  typedef itk::SimpleDataObjectDecorator< RealType >  RealObjectType;
//  typedef SimpleDataObjectDecorator< PixelType > PixelObjectType;

  /** Return the computed Minimum. */
  RealType GetTotalVariation() const
  { return this->GetTotalVariationOutput()->Get(); }
  RealObjectType * GetTotalVariationOutput();

  const RealObjectType * GetTotalVariationOutput() const;

  /** Make a DataObject of the correct type to be used as the specified
   * output. */
  typedef itk::ProcessObject::DataObjectPointerArraySizeType DataObjectPointerArraySizeType;
  using Superclass::MakeOutput;
  DataObjectPointer MakeOutput(DataObjectPointerArraySizeType idx) ITK_OVERRIDE;

#ifdef ITK_USE_CONCEPT_CHECKING
  // Begin concept checking
  itkConceptMacro( InputHasNumericTraitsCheck,
                   ( itk::Concept::HasNumericTraits< PixelType > ) );
  // End concept checking
#endif

  /** Use the image spacing information in calculations. Use this option if you
   *  want derivatives in physical space. Default is UseImageSpacingOn. */
  void SetUseImageSpacingOn()
  { this->SetUseImageSpacing(true); }

  /** Ignore the image spacing. Use this option if you want derivatives in
      isotropic pixel space.  Default is UseImageSpacingOn. */
  void SetUseImageSpacingOff()
  { this->SetUseImageSpacing(false); }

  /** Set/Get whether or not the filter will use the spacing of the input
      image in its calculations */
  itkSetMacro(UseImageSpacing, bool);
  itkGetConstMacro(UseImageSpacing, bool);

protected:
  TotalVariationImageFilter();
  ~TotalVariationImageFilter() {}
  void PrintSelf(std::ostream & os, itk::Indent indent) const ITK_OVERRIDE;

  /** Pass the input through unmodified. Do this by Grafting in the
   *  AllocateOutputs method.
   */
  void AllocateOutputs() ITK_OVERRIDE;

  /** Initialize some accumulators before the threads run. */
  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  /** Do final mean and variance computation from data accumulated in threads.
   */
  void AfterThreadedGenerateData() ITK_OVERRIDE;

  /** Multi-thread version GenerateData. */
  void  ThreadedGenerateData(const RegionType &
                             outputRegionForThread,
                             itk::ThreadIdType threadId) ITK_OVERRIDE;

  // Override since the filter needs all the data for the algorithm
  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  // Override since the filter produces all of its output
  void EnlargeOutputRequestedRegion(itk::DataObject *data) ITK_OVERRIDE;

  bool                        m_UseImageSpacing;

private:
  TotalVariationImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);        //purposely not implemented

  itk::Array< RealType >       m_SumOfSquareRoots;
}; // end of class
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkTotalVariationImageFilter.hxx"
#endif

#endif
