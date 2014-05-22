#ifndef __rtkInterpolatorWithKnownWeightsImageFilter_h
#define __rtkInterpolatorWithKnownWeightsImageFilter_h

#include "itkInPlaceImageFilter.h"
#include "itkArray2D.h"
#include "rtkConfiguration.h"

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
    typedef itk::ImageToImageFilter< VolumeType, VolumeType > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(InterpolatorWithKnownWeightsImageFilter, itk::ImageToImageFilter)

    /** The 3D image to be updated.*/
    void SetInputVolume(const VolumeType* Volume);

    /** The 4D image that will be interpolated, with coefficients, generate a 3D volume.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** Macros that take care of implementing the Get and Set methods for Weights and ProjectionNumber.*/
    itkGetMacro(Weights, itk::Array2D<float>)
    itkSetMacro(Weights, itk::Array2D<float>)

    itkGetMacro(ProjectionNumber, int)
    itkSetMacro(ProjectionNumber, int)

protected:
    InterpolatorWithKnownWeightsImageFilter();
    ~InterpolatorWithKnownWeightsImageFilter(){}

    typename VolumeType::ConstPointer GetInputVolume();
    typename VolumeSeriesType::Pointer GetInputVolumeSeries();

    /** Does the real work. */
    virtual void ThreadedGenerateData( const typename VolumeType::RegionType& outputRegionForThread, ThreadIdType threadId );

    itk::Array2D<float> m_Weights;
    int m_ProjectionNumber;

private:
    InterpolatorWithKnownWeightsImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented


};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkInterpolatorWithKnownWeightsImageFilter.txx"
#endif

#endif
