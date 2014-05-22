#ifndef __rtkFourDROOSTERConeBeamReconstructionFilter_h
#define __rtkFourDROOSTERConeBeamReconstructionFilter_h

#include <itkThresholdImageFilter.h>

#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkAverageOutOfROIImageFilter.h"
#include "rtkTotalVariationDenoisingBPDQImageFilter.h"

namespace rtk
{
  /** \class FourDROOSTERConeBeamReconstructionFilter
   * \brief Implements 4D RecOnstructiOn using Spatial and TEmporal
   * Regularization (short 4D ROOSTER)
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
   * and then applying several regularization steps :
   * - Replacing all negative values by zero
   * - Averaging along time where no movement is expected
   * - Applying total variation denoising in space
   * - Applying total variation denoising in time
   *
   * and starting over.
   *
   * \dot
   * digraph FourDROOSTERConeBeamReconstructionFilter {
   *
   * Input0 [ label="Input 0 (Input: 4D sequence of volumes)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * FourDCG [ label="rtk::FourDConjugateGradientConeBeamReconstructionFilter" URL="\ref rtk::FourDConjugateGradientConeBeamReconstructionFilter"];
   * Positivity [ label="itk::ThresholdImageFilter (positivity)" URL="\ref itk::ThresholdImageFilter"];
   * ROI [ label="rtk::AverageOutOfROIImageFilter" URL="\ref rtk::AverageOutOfROIImageFilter"];
   * TVSpace [ label="rtk::TotalVariationDenoisingBPDQImageFilter (along space)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   * TVTime [ label="rtk::TotalVariationDenoisingBPDQImageFilter (along time)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   * AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterTVTime [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * Input0 -> AfterInput0 [arrowhead=None];
   * AfterInput0 -> FourDCG;
   * Input1 -> FourDCG;
   * FourDCG -> Positivity;
   * Positivity -> ROI;
   * ROI -> TVSpace;
   * TVSpace -> TVTime;
   * TVTime -> AfterTVTime [arrowhead=None];
   * AfterTVTime -> Output;
   * AfterTVTime -> AfterInput0 [style=dashed];
   * }
   * \enddot
   *
   * \test rtkfourdroostertest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename VolumeSeriesType, typename ProjectionStackType>
class FourDROOSTERConeBeamReconstructionFilter : public rtk::IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
{
public:
    /** Standard class typedefs. */
    typedef FourDROOSTERConeBeamReconstructionFilter             Self;
    typedef rtk::IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;
    typedef ProjectionStackType             VolumeType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(FourDROOSTERConeBeamReconstructionFilter, itk::ImageToImageFilter)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** The stack of measured projections */
    void SetInputProjectionStack(const ProjectionStackType* Projection);

    /** The stack of measured projections */
    void SetInputROI(const VolumeType* ROI);

    typedef rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>  FourDCGFilterType;
    typedef itk::ThresholdImageFilter<VolumeSeriesType>                                                     ThresholdFilterType;
    typedef rtk::AverageOutOfROIImageFilter <VolumeSeriesType>                                              AverageOutOfROIFilterType;
    typedef rtk::TotalVariationDenoisingBPDQImageFilter<VolumeSeriesType>                                   TVDenoisingFilterType;

    /** Pass the ForwardProjection filter to SingleProjectionToFourDFilter */
    void SetForwardProjectionFilter(int fwtype);

    /** Pass the backprojection filter to ProjectionStackToFourD*/
    void SetBackProjectionFilter(int bptype);

    /** Pass the interpolation weights to SingleProjectionToFourDFilter */
    void SetWeights(const itk::Array2D<float> _arg);

    /** Pass the motion weights to the gradient and divergence filters*/
    void SetROI(VolumeType* ROI);

    void PrintTiming(std::ostream& os) const;

    // Main loop iterations
    itkSetMacro(MainLoop_iterations, int)
    itkGetMacro(MainLoop_iterations, int)

    // Conjugate gradient iterations
    itkSetMacro(CG_iterations, int)
    itkGetMacro(CG_iterations, int)

    // TV filter parameters
    itkSetMacro(GammaSpace, float)
    itkGetMacro(GammaSpace, float)

    itkSetMacro(GammaTime, float)
    itkGetMacro(GammaTime, float)

    itkSetMacro(TV_iterations, int)
    itkGetMacro(TV_iterations, int)

    itkSetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer)
    itkGetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer)

protected:
    FourDROOSTERConeBeamReconstructionFilter();
    ~FourDROOSTERConeBeamReconstructionFilter(){}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename ProjectionStackType::Pointer   GetInputProjectionStack();
    typename VolumeType::Pointer            GetInputROI();

    /** Does the real work. */
    virtual void GenerateData();

    virtual void GenerateOutputInformation();

    /** Member pointers to the filters used internally (for convenience)*/
    typename FourDCGFilterType::Pointer                     m_FourDCGFilter;
    typename ThresholdFilterType::Pointer                   m_PositivityFilter;
    typename AverageOutOfROIFilterType::Pointer             m_AverageOutOfROIFilter;
    typename TVDenoisingFilterType::Pointer                 m_TVDenoisingSpace;
    typename TVDenoisingFilterType::Pointer                 m_TVDenoisingTime;

private:
    FourDROOSTERConeBeamReconstructionFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    // Regularization parameters
    float m_GammaSpace;
    float m_GammaTime;
    bool m_DimensionsProcessedForTVSpace[VolumeSeriesType::ImageDimension];
    bool m_DimensionsProcessedForTVTime[VolumeSeriesType::ImageDimension];

    // Iterations
    int   m_MainLoop_iterations;
    int   m_CG_iterations;
    int   m_TV_iterations;

    // Geometry
    typename rtk::ThreeDCircularProjectionGeometry::Pointer m_Geometry;

    /** Time probes */
    itk::TimeProbe m_CGProbe;
    itk::TimeProbe m_PositivityProbe;
    itk::TimeProbe m_ROIProbe;
    itk::TimeProbe m_TVSpaceProbe;
    itk::TimeProbe m_TVTimeProbe;
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDROOSTERConeBeamReconstructionFilter.txx"
#endif

#endif
