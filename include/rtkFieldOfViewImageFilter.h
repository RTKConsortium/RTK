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

#ifndef rtkFieldOfViewImageFilter_h
#define rtkFieldOfViewImageFilter_h

#include <itkInPlaceImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConfiguration.h"

struct _lprec;

namespace rtk
{

/** \class FieldOfViewImageFilter
 * \brief Computes the field of view mask for circular 3D geometry.
 *
 * Masks out the regions that are not included in our field of view or
 * creates the mask if m_Mask is true. Note that the 3 angle parameters are
 * assumed to be 0. in the circular geometry: GantryAngle, OutOfPlaneAngle and
 * InPlaneAngle. The rest is accounted for but the fov is assumed to be
 * cylindrical.
 *
 * \test rtkfovtest.cxx, rtkfdktest.cxx, rtkmotioncompensatedfdktest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT FieldOfViewImageFilter:
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FieldOfViewImageFilter                            Self;
  typedef itk::ImageToImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  typedef typename TOutputImage::RegionType                 OutputImageRegionType;
  typedef typename TInputImage::Superclass::Pointer         ProjectionsStackPointer;
  typedef rtk::ThreeDCircularProjectionGeometry             GeometryType;
  typedef typename GeometryType::Pointer                    GeometryPointer;
  typedef enum {RADIUSINF,RADIUSSUP,RADIUSBOTH}             FOVRadiusType;


  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FieldOfViewImageFilter, InPlaceImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Get / Set of the member Mask. If set, all the pixels in the field of view
   * are set to 1. The data value is kept otherwise. Pixels outside the mask
   * are set to 0 in any case. */
  itkGetMacro(Mask, bool);
  itkSetMacro(Mask, bool);

  /** Get / Set the region of projection images, required to determine the
    FOV radius. Note that only the geometric information is required, the
    data are therefore not updated. */
  itkGetMacro(ProjectionsStack, ProjectionsStackPointer);
  itkSetMacro(ProjectionsStack, ProjectionsStackPointer);

  /** Assume that a displaced detector image filter, e.g.,
   * rtk::DisplacedDetectorImageFilter, has been used. */
  itkGetMacro(DisplacedDetector, bool);
  itkSetMacro(DisplacedDetector, bool);

  /** Computes the radius r and the center (x,z) of the disk perpendicular to
   * the y-axis that is covered by:
   * - if RADIUSINF: the half plane defined by the line from the source to the
   *   two inferior x index corners which cover the opposite two corners.
   * - if RADIUSSUP: the half plane defined by the line from the source to the
   *   two superior x index corners which cover the opposite two corners.
   * - if RADIUSBOTH: the fan defined by the pairs of half planes.
   * Returns true if it managed to find such a disk and false otherwise.
   * The function may be called without out computing the output, but
   * m_Geometry and ProjectionsStack must be set.*/
  virtual bool ComputeFOVRadius(const FOVRadiusType type, double &x, double &z, double &r);

  /** Add collimation constraints. This function is always called from
   * ComputeFOVRadius but it has an effect only if the geometry has the
   * m_CollimationUInf or m_CollimationUSup which are non infinity (default). */
  void AddCollimationConstraints(const FOVRadiusType type, _lprec *lp);

protected:
  FieldOfViewImageFilter();
  ~FieldOfViewImageFilter() {}

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  /** Generates a FOV mask which is applied to the reconstruction
   * A call to this function will assume modification of the function.*/
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

private:
  FieldOfViewImageFilter(const Self&);      //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  GeometryPointer         m_Geometry;
  bool                    m_Mask;
  ProjectionsStackPointer m_ProjectionsStack;
  double                  m_Radius;
  double                  m_CenterX;
  double                  m_CenterZ;
  double                  m_HatTangentInf;
  double                  m_HatTangentSup;
  double                  m_HatHeightInf;
  double                  m_HatHeightSup;
  bool                    m_DisplacedDetector;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFieldOfViewImageFilter.hxx"
#endif

#endif
