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
#ifndef rtkSplatWithKnownWeightsImageFilter_h
#define rtkSplatWithKnownWeightsImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkArray2D.h>

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  #include <itkImageRegionSplitterDirection.h>
#endif

#include "rtkMacro.h"

namespace rtk
{
  /** \class SplatWithKnownWeightsImageFilter
   * \brief Splats (linearly) a 3D volume into a 3D+t sequence of volumes
   *
   * See the reference paper: "Cardiac C-arm computed tomography using
   * a 3D + time ROI reconstruction method with spatial and temporal regularization"
   * by Mory et al.
   *
   * 4D conjugate gradient reconstruction consists in iteratively
   * minimizing the following cost function:
   *
   * Sum_over_theta || R_theta S_theta f - p_theta ||_2^2
   *
   * with
   * - f a 4D series of 3D volumes, each one being the reconstruction
   * at a given respiratory/cardiac phase
   * - p_theta is the projection measured at angle theta
   * - S_theta an interpolation operator which, from the 3D + time sequence f,
   * estimates the 3D volume through which projection p_theta has been acquired
   * - R_theta is the X-ray transform (the forward projection operator) for angle theta
   *
   * Computing the gradient of this cost function yields:
   *
   * S_theta^T R_theta^T R_theta S_theta f - S_theta^T R_theta^T p_theta
   *
   * where A^T means the adjoint of operator A.
   *
   * SplatWithKnownWeightsImageFilter implements S_theta^T.
   *
   *
   * \test rtkfourdconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename VolumeSeriesType, typename VolumeType>
class SplatWithKnownWeightsImageFilter : public itk::InPlaceImageFilter< VolumeSeriesType, VolumeSeriesType >
{
public:
    /** Standard class typedefs. */
    typedef SplatWithKnownWeightsImageFilter                              Self;
    typedef itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType > Superclass;
    typedef itk::SmartPointer< Self >                                     Pointer;
    typedef typename VolumeSeriesType::RegionType                         OutputImageRegionType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(SplatWithKnownWeightsImageFilter, itk::ImageToImageFilter)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** The 3D image that will be added, with coefficients, to each 3D volume of the 4D image.*/
    void SetInputVolume(const VolumeType* Volume);

    /** Macros that take care of implementing the Get and Set methods for Weights and projectionNumber.*/
    itkGetMacro(Weights, itk::Array2D<float>)
    itkSetMacro(Weights, itk::Array2D<float>)

    itkGetMacro(ProjectionNumber, int)
    void SetProjectionNumber(int n);

protected:
    SplatWithKnownWeightsImageFilter();
    ~SplatWithKnownWeightsImageFilter() {}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename VolumeType::Pointer GetInputVolume();

    /** Does the real work. */
    void ThreadedGenerateData(const typename VolumeSeriesType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;
    void BeforeThreadedGenerateData() ITK_OVERRIDE;

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
    /** Splits the OutputRequestedRegion along the first direction, not the last */
    const itk::ImageRegionSplitterBase* GetImageRegionSplitter(void) const ITK_OVERRIDE;
    itk::ImageRegionSplitterDirection::Pointer  m_Splitter;
#endif

    itk::Array2D<float>                         m_Weights;
    int                                         m_ProjectionNumber;


private:
    SplatWithKnownWeightsImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSplatWithKnownWeightsImageFilter.hxx"
#endif

#endif
