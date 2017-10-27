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

#ifndef rtkDisplacedDetectorImageFilter_h
#define rtkDisplacedDetectorImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class DisplacedDetectorImageFilter
 * \brief Weigting for displaced detectors
 *
 * Weighting of image projections to handle off-centered panels
 * in tomography reconstruction. Based on [Wang, Med Phys, 2002].
 *
 * Note that the filter does nothing if the panel shift is less than 10%
 * of its size. Otherwise, it does the weighting described in the publication
 * and zero pads the data on the nearest side to the center.
 * Therefore, the InPlace capability depends on the displacement.
 * It can only be inplace if there is no displacement, it can not otherwise.
 * The GenerateOutputInformation method takes care of properly setting this up.
 *
 * By default, it computes the minimum and maximum offsets from the complete geometry object.
 * When an independent projection has to be processed, these values have to be set by the user from
 * a priori knowledge of the detector displacements.
 *
 * The weighting accounts for variations in SourceToDetectorDistances,
 * SourceOffsetsX and ProjectionOffsetsX. It currently assumes constant
 * SourceToIsocenterDistances and 0. InPlaneAngles. The other parameters are
 * not relevant in the computation because the weighting is reproduced at
 * every gantry angle on each line of the projection images.
 *
 * \test rtkdisplaceddetectortest.cxx, rtkdisplaceddetectorcompcudatest.cxx,
 * rtkdisplaceddetectorcompoffsettest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT DisplacedDetectorImageFilter :
  public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DisplacedDetectorImageFilter Self;

  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;

  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

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
  itkTypeMacro(DisplacedDetectorImageFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Get / Set whether the projections should be padded
   * (yes for FDK, no for iterative) */
  itkGetMacro(PadOnTruncatedSide, bool);
  itkSetMacro(PadOnTruncatedSide, bool);

  /**
   * Get / Set the minimum and maximum offsets of the detector along the
   * weighting direction desribed in ToUntiltedCoordinate.
   */
  void SetOffsets(double minOffset, double maxOffset);
  itkGetMacro(MinimumOffset, double);
  itkGetMacro(MaximumOffset, double);

  /** Get / Set the Disable parameter
   */
  itkGetMacro(Disable, bool);
  itkSetMacro(Disable, bool);

protected:
  DisplacedDetectorImageFilter();

  ~DisplacedDetectorImageFilter() {}

  /** Retrieve computed inferior and superior corners */
  itkGetMacro(InferiorCorner, double);
  itkGetMacro(SuperiorCorner, double);

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

  // Iterative filters do not need padding
  bool m_PadOnTruncatedSide;

private:
  DisplacedDetectorImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);               //purposely not implemented

  /** RTK geometry object */
  GeometryPointer m_Geometry;

  /**
   * Minimum and maximum offsets of the detector along the weighting direction, i.e. x.
   * If a priori known, these values can be given as input. Otherwise, they are computed from the
   * complete geometry.
   */
  double m_MinimumOffset;
  double m_MaximumOffset;

  /**
   * Flag used to know if the user has entered the min/max values of the detector offset.
   */
  bool m_OffsetsSet;

  /** Superior and inferior position of the detector along the weighting
   *  direction, i.e., the virtual detector described in ToUntiltedCoordinate.
   */
  double m_InferiorCorner;
  double m_SuperiorCorner;

  /** When using a geometry that the displaced detector cannot manage,
   * it has to be disabled
   */
  bool m_Disable;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDisplacedDetectorImageFilter.hxx"
#endif

#endif
