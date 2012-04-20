#ifndef __rtkFDKWeightProjectionFilter_h
#define __rtkFDKWeightProjectionFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

/** \class FDKWeightProjectionFilter
 * \brief Weighting of projections to correct for the divergence in
 * filtered backprojection reconstruction algorithms.
 * The weighting comprises:
 * - the 2D weighting of the FDK algorithm [Feldkamp, 1984],
 * - the correction of the ramp factor for divergent full scan,
 * - the angular weighting for the final 3D integral of FDK.
 * Note that SourceToDetectorDistance, SourceToDetectorIsocenter
 * SouceOffsets and ProjectionOffsets are accounted for on a per
 * projection basis but InPlaneRotation and OutOfPlaneRotation are not
 * accounted for.
 * \author Simon Rit
 */
namespace rtk
{

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT FDKWeightProjectionFilter :
  public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FDKWeightProjectionFilter Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                          InputImageType;
  typedef TOutputImage                         OutputImageType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(FDKWeightProjectionFilter, itk::ImageToImageFilter);

  /** Get/ Set geometry structure */
  itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);

protected:
  FDKWeightProjectionFilter()  {}
  ~FDKWeightProjectionFilter() {}

  virtual void BeforeThreadedGenerateData();

  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId);

private:
  FDKWeightProjectionFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** Angular weights for each projection */
  std::vector<double> m_AngularWeightsAndRampFactor;

  /** Geometrical description of the system */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFDKWeightProjectionFilter.txx"
#endif

#endif
