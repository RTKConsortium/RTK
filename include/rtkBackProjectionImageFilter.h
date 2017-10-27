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

#ifndef rtkBackProjectionImageFilter_h
#define rtkBackProjectionImageFilter_h

#include "rtkConfiguration.h"

#include <itkInPlaceImageFilter.h>
#include <itkConceptChecking.h>
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class BackProjectionImageFilter
 * \brief 3D backprojection
 *
 * Backprojects a stack of projection images (input 1) in a 3D volume (input 0)
 * using linear interpolation according to a specified geometry. The operation
 * is voxel-based, meaning that the center of each voxel is projected in the
 * projection images to determine the interpolation location.
 *
 * \test rtkfovtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup Projector
 */
template <class TInputImage, class TOutputImage>
class BackProjectionImageFilter :
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

  typedef rtk::ThreeDCircularProjectionGeometry                     GeometryType;
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
  BackProjectionImageFilter() : m_Geometry(ITK_NULLPTR), m_Transpose(false) {
    this->SetNumberOfRequiredInputs(2); this->SetInPlace( true );
  };
  ~BackProjectionImageFilter() {}

  /** Apply changes to the input image requested region. */
  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

  /** Special case when the detector is cylindrical and centered on source */
  virtual void CylindricalDetectorCenteredOnSourceBackprojection(const OutputImageRegionType& region,
                                                                 const ProjectionMatrixType& volIndexToProjPP,
                                                                 const itk::Matrix<double, TInputImage::ImageDimension, TInputImage::ImageDimension>& projPPToProjIndex,
                                                                 const ProjectionImagePointer projection);

  /** Optimized version when the rotation is parallel to X, i.e. matrix[1][0]
    and matrix[2][0] are zeros. */
  virtual void OptimizedBackprojectionX(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
                                        const ProjectionImagePointer projection);

  /** Optimized version when the rotation is parallel to Y, i.e. matrix[1][1]
    and matrix[2][1] are zeros. */
  virtual void OptimizedBackprojectionY(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
                                        const ProjectionImagePointer projection);

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  /** The input is a stack of projections, we need to interpolate in one projection
      for efficiency during interpolation. Use of itk::ExtractImageFilter is
      not threadsafe in ThreadedGenerateData, this one is. The output can be multiplied by a constant.
      The function is templated to allow getting an itk::CudaImage. */
  template<class TProjectionImage>
  typename TProjectionImage::Pointer GetProjection(const unsigned int iProj);

  /** Creates iProj index to index projection matrices with current inputs
      instead of the physical point to physical point projection matrix provided by Geometry */
  ProjectionMatrixType GetIndexToIndexProjectionMatrix(const unsigned int iProj);

  ProjectionMatrixType GetVolumeIndexToProjectionPhysicalPointMatrix(const unsigned int iProj);

  itk::Matrix<double, TInputImage::ImageDimension, TInputImage::ImageDimension> GetProjectionPhysicalPointToProjectionIndexMatrix();

  /** RTK geometry object */
  GeometryPointer m_Geometry;

private:
  BackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** Flip projection flag: infludences GetProjection and
    GetIndexToIndexProjectionMatrix for optimization */
  bool m_Transpose;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkBackProjectionImageFilter.hxx"
#endif

#endif
