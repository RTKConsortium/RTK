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

#ifndef rtkFDKBackProjectionImageFilter_h
#define rtkFDKBackProjectionImageFilter_h

#include "rtkBackProjectionImageFilter.h"

namespace rtk
{

/** \class FDKBackProjectionImageFilter
 * \brief CPU version of the backprojection of the FDK algorithm.
 *
 * CPU implementation of the backprojection step of the
 * [Feldkamp, Davis, Kress, 1984] algorithm for filtered backprojection
 * reconstruction of cone-beam CT images with a circular source trajectory.
 *
 * \author Simon Rit
 *
 * \ingroup Projector
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT FDKBackProjectionImageFilter :
  public BackProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FDKBackProjectionImageFilter                        Self;
  typedef BackProjectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  typedef typename Superclass::ProjectionMatrixType ProjectionMatrixType;
  typedef typename TOutputImage::RegionType         OutputImageRegionType;
  typedef typename Superclass::ProjectionImageType  ProjectionImageType;
  typedef typename ProjectionImageType::Pointer     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FDKBackProjectionImageFilter, ImageToImageFilter);

protected:
  FDKBackProjectionImageFilter() {};
  ~FDKBackProjectionImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

  /** Optimized version when the rotation is parallel to X, i.e. matrix[1][0]
    and matrix[2][0] are zeros. */
  void OptimizedBackprojectionX(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
                                        const ProjectionImagePointer projection) ITK_OVERRIDE;

  /** Optimized version when the rotation is parallel to Y, i.e. matrix[1][1]
    and matrix[2][1] are zeros. */
  void OptimizedBackprojectionY(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
                                        const ProjectionImagePointer projection) ITK_OVERRIDE;

private:
  FDKBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);               //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFDKBackProjectionImageFilter.hxx"
#endif

#endif
