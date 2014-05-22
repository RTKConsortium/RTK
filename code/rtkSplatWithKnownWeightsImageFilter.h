#ifndef __rtkSplatWithKnownWeightsImageFilter_h
#define __rtkSplatWithKnownWeightsImageFilter_h

#include "itkInPlaceImageFilter.h"

#include "itkArray2D.h"

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
    typedef SplatWithKnownWeightsImageFilter             Self;
    typedef itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;
    typedef typename VolumeSeriesType::RegionType    OutputImageRegionType;

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
    itkSetMacro(ProjectionNumber, int)

protected:
    SplatWithKnownWeightsImageFilter();
    ~SplatWithKnownWeightsImageFilter(){}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename VolumeType::Pointer GetInputVolume();

    /** Does the real work. */
    virtual void ThreadedGenerateData(const typename VolumeSeriesType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId));

    /** Splats the OutputRequestedRegion along the first direction, not the last*/
    unsigned int SplitRequestedRegion(unsigned int i, unsigned int num, typename VolumeSeriesType::RegionType &splatRegion);

    itk::Array2D<float> m_Weights;
    int m_ProjectionNumber;
private:
    SplatWithKnownWeightsImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented



};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSplatWithKnownWeightsImageFilter.txx"
#endif

#endif
