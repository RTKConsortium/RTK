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

#ifndef __rtkBackProjectionImageFilter_h
#define __rtkBackProjectionImageFilter_h

#include "rtkConfiguration.h"

#include <itkInPlaceImageFilter.h>
#include <itkConceptChecking.h>
#include "rtkProjectionGeometry.h"

/** \class BackProjectionImageFilter
 * \brief TODO
 *
 * TODO
 *
 * \author Simon Rit
 *
 * \ingroup InPlaceImageFilter
 */

namespace rtk
{

template <class TInputImage, class TOutputImage>
class ITK_EXPORT BackProjectionImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef BackProjectionImageFilter                         Self;
  typedef itk::ImageToImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TInputImage::PixelType                   InputPixelType;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;

  typedef rtk::ProjectionGeometry<TOutputImage::ImageDimension>     GeometryType;
  typedef typename GeometryType::Pointer                            GeometryPointer;
  typedef typename GeometryType::MatrixType                         ProjectionMatrixType;
  typedef itk::Image<InputPixelType, TInputImage::ImageDimension-1> ProjectionImageType;
  typedef typename ProjectionImageType::Pointer                     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BackProjectionImageFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Get / Set the transpose flag for 2D projections (optimization trick) */
  itkGetMacro(Transpose, bool);
  itkSetMacro(Transpose, bool);

protected:
  BackProjectionImageFilter() : m_Geometry(NULL), m_Transpose(false) {
    this->SetNumberOfRequiredInputs(2); this->SetInPlace( true );
  };
  virtual ~BackProjectionImageFilter() {
  }

  /** Apply changes to the input image requested region. */
  virtual void GenerateInputRequestedRegion();

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

  /** The input is a stack of projections, we need to interpolate in one projection
      for efficiency during interpolation. Use of itk::ExtractImageFilter is
      not threadsafe in ThreadedGenerateData, this one is. The output can be multiplied by a constant. */
  ProjectionImagePointer GetProjection(const unsigned int iProj);

  /** Creates the #iProj index to index projection matrix with current inputs
      instead of the physical point to physical point projection matrix provided by Geometry */
  ProjectionMatrixType GetIndexToIndexProjectionMatrix(const unsigned int iProj, const ProjectionImageType *proj);

private:
  BackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** RTK geometry object */
  GeometryPointer m_Geometry;

  /** Flip projection flag: infludences GetProjection and
    GetIndexToIndexProjectionMatrix for optimization */
  bool m_Transpose;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkBackProjectionImageFilter.txx"
#endif

#endif
