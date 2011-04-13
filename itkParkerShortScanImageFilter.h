#ifndef __itkParkerShortScanImageFilter_h
#define __itkParkerShortScanImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "itkThreeDCircularProjectionGeometry.h"

/** \class ParkerShortScanImageFilter
 *
 * Weighting of image projections to handle off-centered panels
 * in tomography reconstruction. Based on [Wang, Med Phys, 2002].
 *
 * Note that the filter does nothing if the panel shift is less than 10%
 * of its size. Otherwise, it does the weighting described in the publication
 * and zero pads the data on the nearest side to the center.
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT ParkerShortScanImageFilter :
  public InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ParkerShortScanImageFilter Self;

  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;

  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                     InputImageType;
  typedef TOutputImage                                    OutputImageType;
  typedef typename OutputImageType::RegionType            OutputImageRegionType;
  typedef itk::Image<typename TOutputImage::PixelType, 1> WeightImageType;

  typedef ThreeDCircularProjectionGeometry GeometryType;
  typedef GeometryType::Pointer            GeometryPointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ParkerShortScanImageFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

protected:
  ParkerShortScanImageFilter(){ this->SetInPlace(true); }
  ~ParkerShortScanImageFilter(){}

  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId);

private:
  ParkerShortScanImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);             //purposely not implemented

  /** RTK geometry object */
  GeometryPointer m_Geometry;

  /** Superior and inferior position of the detector along the weighting direction, i.e. x.
   * The computed value account for the x projection offset of the geometry.
   */
  double m_InferiorCorner;
  double m_SuperiorCorner;
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkParkerShortScanImageFilter.txx"
#endif

#endif
