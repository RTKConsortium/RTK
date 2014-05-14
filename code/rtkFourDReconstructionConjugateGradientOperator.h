#ifndef __rtkFourDReconstructionConjugateGradientOperator_h
#define __rtkFourDReconstructionConjugateGradientOperator_h

#include "rtkConjugateGradientOperator.h"

#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

#include "rtkProjectionStackToFourDImageFilter.h"
#include "rtkFourDToProjectionStackImageFilter.h"
#include "rtkConstantImageSource.h"

#include "rtkThreeDCircularProjectionGeometry.h"
#include "itkArray2D.h"

namespace rtk
{
template< typename VolumeSeriesType, typename ProjectionStackType>
class FourDReconstructionConjugateGradientOperator : public ConjugateGradientOperator< VolumeSeriesType>
{
public:
    /** Standard class typedefs. */
    typedef FourDReconstructionConjugateGradientOperator        Self;
    typedef ConjugateGradientOperator< VolumeSeriesType>        Superclass;
    typedef itk::SmartPointer< Self >                           Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(FourDReconstructionConjugateGradientOperator, ConjugateGradientOperator)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** The image that will be backprojected, then added, with coefficients, to each 3D volume of the 4D image.
    * It is 3D because the backprojection filters need it, but the third dimension, which is the number of projections, is 1  */
    void SetInputProjectionStack(const ProjectionStackType* Projection);

    typedef rtk::BackProjectionImageFilter< ProjectionStackType, ProjectionStackType >          BackProjectionFilterType;
    typedef rtk::ForwardProjectionImageFilter< ProjectionStackType, ProjectionStackType >       ForwardProjectionFilterType;
    typedef rtk::FourDToProjectionStackImageFilter< ProjectionStackType, VolumeSeriesType >     FourDToProjectionStackFilterType;
    typedef rtk::ProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType >     ProjectionStackToFourDFilterType;
    typedef rtk::ConstantImageSource<VolumeSeriesType>                                          ConstantImageSourceFilterType;

    /** Pass the backprojection filter to ProjectionStackToFourD*/
    void SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg);

    /** Pass the forward projection filter to FourDToProjectionStack */
    void SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg);

    /** Pass the geometry to both ProjectionStackToFourD and FourDToProjectionStack */
    void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

    /** Pass the interpolation weights to both ProjectionStackToFourD and FourDToProjectionStack */
    void SetWeights(const itk::Array2D<float> _arg);


protected:
    FourDReconstructionConjugateGradientOperator();
    ~FourDReconstructionConjugateGradientOperator(){}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename ProjectionStackType::ConstPointer GetInputProjectionStack();

    /** Builds the pipeline and computes output information */
    virtual void GenerateOutputInformation();

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename FourDToProjectionStackFilterType::Pointer     m_FourDToProjectionStackFilter;
    typename ProjectionStackToFourDFilterType::Pointer     m_ProjectionStackToFourDFilter;
    typename ConstantImageSourceFilterType::Pointer        m_ConstantImageSourceFilter;

private:
    FourDReconstructionConjugateGradientOperator(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDReconstructionConjugateGradientOperator.txx"
#endif

#endif
