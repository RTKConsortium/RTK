#ifndef __itkFDKWeightProjectionFilter_h
#define __itkFDKWeightProjectionFilter_h

#include "itkInPlaceImageFilter.h"
#include "itkThreeDCircularProjectionGeometry.h"

/** \class FDKWeightProjectionFilter
 * \brief Weighting of projections to correct for the divergence in
 * filtered backprojection reconstruction algorithms.
 * The weighting comprises:
 * - the 2D weighting of the FDK algorithm [Feldkamp, 1984],
 * - the correction of the ramp factor for divergent full scan,
 * - the angular weighting for the final 3D integral of FDK.
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT FDKWeightProjectionFilter :
  public InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FDKWeightProjectionFilter Self;

  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;

  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                                                  InputImageType;
  typedef TOutputImage                                                                 OutputImageType;
  typedef typename OutputImageType::RegionType                                         OutputImageRegionType;
  typedef itk::Image<typename TOutputImage::PixelType, TOutputImage::ImageDimension-1> WeightImageType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(FDKWeightProjectionFilter, ImageToImageFilter);

  /** Get/ Set geometry structure */
  itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);


protected:
  FDKWeightProjectionFilter():m_WeightsImage(NULL){ this->SetInPlace(true); }
  ~FDKWeightProjectionFilter(){}

  virtual void EnlargeOutputRequestedRegion( DataObject *output );

  virtual void BeforeThreadedGenerateData();

  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId);

private:
  FDKWeightProjectionFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** Angular weights for each projection */
  std::vector<double> m_AngularWeights;

  /** Geometrical description of the system */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** One line of weights which exclude the angular weighting */
  typename WeightImageType::Pointer m_WeightsImage;
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFDKWeightProjectionFilter.txx"
#endif

#endif
