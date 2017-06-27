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

#ifndef rtkNormalizedJosephBackProjectionImageFilter_h
#define rtkNormalizedJosephBackProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include <itkAddImageFilter.h>
#include <itkDivideOrZeroOutImageFilter.h>

namespace rtk
{
/** \class NormalizedJosephBackProjectionImageFilter
 * \brief Normalized Joseph back projection.
 *
 * Performs a Jospeh back projection and divides it by the Joseph back projection
 * of a projection filled with ones, to meet the requirements of SART.
 *
 * \dot
 * digraph NormalizedJosephBackProjectionImageFilter {
 *
 * Input0 [ label="Input 0 (Volume)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Output [label="Output (Reconstruction)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * ConstantVolumeSource [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * ConstantProjectionSource [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * BackProjection [ label="rtk::JosephBackProjectionImageFilter" URL="\ref rtk::JosephBackProjectionImageFilter"];
 * BackProjectionOfConstant [ label="rtk::JosephBackProjectionImageFilter" URL="\ref rtk::JosephBackProjectionImageFilter"];
 * Divide [ label="itk::DivideImageFilter" URL="\ref itk::DivideImageFilter"];
 * Add [ label="itk::AddImageFilter (by lambda)" URL="\ref itk::AddImageFilter"];
 * OutofConstantVolumeSource [label="", fixedsize="false", width=0, height=0, shape=none];
 * ConstantVolumeSource -> OutofConstantVolumeSource [arrowhead=none];
 * OutofConstantVolumeSource -> BackProjection [ label="#0"];
 * OutofConstantVolumeSource -> BackProjectionOfConstant [ label="#0"];
 * Input1 -> BackProjection [ label="#1"];
 * ConstantProjectionSource -> BackProjectionOfConstant [ label="#1"];
 * BackProjectionOfConstant -> Divide [ label="#1"];
 * BackProjection -> Divide [ label="#0"];
 * Divide -> Add;
 * Input0 -> Add;
 * Add -> Output;
 * }
 * \enddot
 *
 * \test rtksarttest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup Projector
 */

template <class TInputImage,
          class TOutputImage>
class ITK_EXPORT NormalizedJosephBackProjectionImageFilter :
  public BackProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef NormalizedJosephBackProjectionImageFilter              Self;
  typedef BackProjectionImageFilter<TInputImage,TOutputImage>    Superclass;
  typedef itk::SmartPointer<Self>                                Pointer;
  typedef itk::SmartPointer<const Self>                          ConstPointer;
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TOutputImage::PixelType                       OutputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;
  typedef rtk::ThreeDCircularProjectionGeometry                  GeometryType;
  typedef typename GeometryType::Pointer                         GeometryPointer;

  /** Typedefs for subfilters */
  typedef itk::AddImageFilter<TOutputImage,TOutputImage> AddFilterType;
  typedef itk::DivideOrZeroOutImageFilter<TOutputImage,TOutputImage> DivideFilterType;
  typedef ConstantImageSource<TInputImage>  ConstantProjectionSourceType;
  typedef ConstantImageSource<TOutputImage> ConstantVolumeSourceType;
  typedef JosephBackProjectionImageFilter<TInputImage,TOutputImage> JosephBackProjectionFilterType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(NormalizedJosephBackProjectionImageFilter, BackProjectionImageFilter);

protected:
  NormalizedJosephBackProjectionImageFilter();
  ~NormalizedJosephBackProjectionImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  /** Sub filters */
  typename AddFilterType::Pointer                   m_AddFilter;
  typename DivideFilterType::Pointer                m_DivideFilter;
  typename ConstantProjectionSourceType::Pointer    m_ConstantProjectionSource;
  typename ConstantVolumeSourceType::Pointer        m_ConstantVolumeSource;
  typename JosephBackProjectionFilterType::Pointer  m_JosephBackProjector;
  typename JosephBackProjectionFilterType::Pointer  m_JosephBackProjectorOfConstantProjection;

private:
  NormalizedJosephBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented

};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkNormalizedJosephBackProjectionImageFilter.hxx"
#endif

#endif
