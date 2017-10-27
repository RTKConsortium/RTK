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
#ifndef rtkInterpolatorWithKnownWeightsImageFilter_h
#define rtkInterpolatorWithKnownWeightsImageFilter_h

#include "itkInPlaceImageFilter.h"
#include "itkArray2D.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"

namespace rtk
{
  /** \class InterpolatorWithKnownWeightsImageFilter
   * \brief Interpolates (linearly) in a 3D+t sequence of volumes to get a 3D volume
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
   * InterpolatorWithKnownWeightsImageFilter implements S_theta.
   *
   *
   * \test rtkfourdconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */
template< typename VolumeType, typename VolumeSeriesType>
class InterpolatorWithKnownWeightsImageFilter : public itk::InPlaceImageFilter< VolumeType, VolumeType >
{
public:
    /** Standard class typedefs. */
    typedef InterpolatorWithKnownWeightsImageFilter             Self;
    typedef itk::ImageToImageFilter< VolumeType, VolumeType >   Superclass;
    typedef itk::SmartPointer< Self >                           Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(InterpolatorWithKnownWeightsImageFilter, itk::InPlaceImageFilter)

    /** The 3D image to be updated.*/
    void SetInputVolume(const VolumeType* Volume);

    /** The 4D image that will be interpolated, with coefficients, generate a 3D volume.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** Macros that take care of implementing the Get and Set methods for Weights and ProjectionNumber.*/
    itkGetMacro(Weights, itk::Array2D<float>)
    itkSetMacro(Weights, itk::Array2D<float>)

    itkGetMacro(ProjectionNumber, int)
    void SetProjectionNumber(int n);

protected:
    InterpolatorWithKnownWeightsImageFilter();
    ~InterpolatorWithKnownWeightsImageFilter() {}

    typename VolumeType::ConstPointer GetInputVolume();
    typename VolumeSeriesType::Pointer GetInputVolumeSeries();

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Does the real work. */
    void ThreadedGenerateData( const typename VolumeType::RegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

    itk::Array2D<float> m_Weights;
    int                 m_ProjectionNumber;

private:
    InterpolatorWithKnownWeightsImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented


};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkInterpolatorWithKnownWeightsImageFilter.hxx"
#endif

#endif
