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

#ifndef rtkDrawGeometricPhantomImageFilter_h
#define rtkDrawGeometricPhantomImageFilter_h

#include <itkInPlaceImageFilter.h>

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include "rtkGeometricPhantomFileReader.h"

#include <vector>

namespace rtk
{

/** \class DrawGeometricPhantomImageFilter
 * \brief Draw quadric shapes in 3D image.
 *
 * The filter draws a list of quadric shapes which parameters are passed by a
 * file. See rtkGeometricPhantomFileReader.h for the file format.
 *
 * \test rtkdrawgeometricphantomtest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT DrawGeometricPhantomImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DrawGeometricPhantomImageFilter                           Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage>         Superclass;
  typedef itk::SmartPointer<Self>                                   Pointer;
  typedef itk::SmartPointer<const Self>                             ConstPointer;
  typedef typename TOutputImage::RegionType                         OutputImageRegionType;

  typedef itk::Vector<double,3>                                     VectorType;
  typedef std::vector< std::vector<double> >                        VectorOfVectorType;
  typedef std::string                                               StringType;

  typedef rtk::ConvertEllipsoidToQuadricParametersFunction          EQPFunctionType;
  typedef rtk::GeometricPhantomFileReader                           CFRType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawGeometricPhantomImageFilter, InPlaceImageFilter);

  /** Get/Set ConfigFile*/
  itkSetMacro(ConfigFile, StringType);
  itkGetMacro(ConfigFile, StringType);

  /** Multiplicative Scaling factor for the phantom described ConfigFile. */
  itkSetMacro(PhantomScale, VectorType);
  itkGetMacro(PhantomScale, VectorType);

  /** Get / Set the spatial position of the phantom Shepp Logan phantom relative to its
   * center. The default value is (0, 0, 0). */
  itkSetMacro(OriginOffset, VectorType);
  itkGetMacro(OriginOffset, VectorType);

protected:
  DrawGeometricPhantomImageFilter();
  ~DrawGeometricPhantomImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

private:
  DrawGeometricPhantomImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented
  StringType m_ConfigFile;

  VectorType m_PhantomScale;
  VectorType m_OriginOffset;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawGeometricPhantomImageFilter.hxx"
#endif

#endif
