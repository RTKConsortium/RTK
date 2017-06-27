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

#ifndef rtkFDKConeBeamReconstructionFilter_h
#define rtkFDKConeBeamReconstructionFilter_h

#include "rtkFDKWeightProjectionFilter.h"
#include "rtkFFTRampImageFilter.h"
#include "rtkFDKBackProjectionImageFilter.h"
#include "rtkConfiguration.h"

#include <itkExtractImageFilter.h>
#include <itkTimeProbe.h>

namespace rtk
{

/** \class FDKConeBeamReconstructionFilter
 * \brief Implements Feldkamp, David and Kress cone-beam reconstruction
 *
 * FDKConeBeamReconstructionFilter is a mini-pipeline filter which combines
 * the different steps of the FDK cone-beam reconstruction filter:
 * - rtk::FDKWeightProjectionFilter for 2D weighting of the projections,
 * - rtk::FFTRampImageFilter for ramp filtering,
 * - rtk::FDKBackProjectionImageFilter for backprojection.
 * The input stack of projections is processed piece by piece (the size is
 * controlled with ProjectionSubsetSize) via the use of itk::ExtractImageFilter
 * to extract sub-stacks.
 *
 * \dot
 * digraph FDKConeBeamReconstructionFilter {
 * node [shape=box];
 * 1 [ label="rtk::FDKWeightProjectionFilter" URL="\ref rtk::FDKWeightProjectionFilter"];
 * 2 [ label="rtk::FFTRampImageFilter" URL="\ref rtk::FFTRampImageFilter"];
 * 3 [ label="rtk::FDKBackProjectionImageFilter" URL="\ref rtk::FDKBackProjectionImageFilter"];
 * 1 -> 2;
 * 2 -> 3;
 * }
 * \enddot
 *
 * \test rtkfdktest.cxx, rtkrampfiltertest.cxx, rtkmotioncompensatedfdktest.cxx,
 * rtkdisplaceddetectortest.cxx, rtkshortscantest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ReconstructionAlgorithm
 */
template<class TInputImage, class TOutputImage=TInputImage, class TFFTPrecision=double>
class ITK_EXPORT FDKConeBeamReconstructionFilter :
  public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FDKConeBeamReconstructionFilter                    Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Typedefs of each subfilter of this composite filter */
  typedef itk::ExtractImageFilter<InputImageType, OutputImageType>                 ExtractFilterType;
  typedef rtk::FDKWeightProjectionFilter<InputImageType, OutputImageType>          WeightFilterType;
  typedef rtk::FFTRampImageFilter<OutputImageType, OutputImageType, TFFTPrecision> RampFilterType;
  typedef rtk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType>      BackProjectionFilterType;
  typedef typename BackProjectionFilterType::Pointer                               BackProjectionFilterPointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(FDKConeBeamReconstructionFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  virtual ThreeDCircularProjectionGeometry::Pointer GetGeometry();
  virtual void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

  /** Get pointer to the weighting filter used by the feldkamp reconstruction */
  typename WeightFilterType::Pointer GetWeightFilter() { return m_WeightFilter; }

  /** Get pointer to the ramp filter used by the feldkamp reconstruction */
  typename RampFilterType::Pointer GetRampFilter() { return m_RampFilter; }

  void PrintTiming(std::ostream& os) const;

  /** Get / Set the number of cone-beam projection images processed
      simultaneously. Default is 4. */
  itkGetMacro(ProjectionSubsetSize, unsigned int);
  itkSetMacro(ProjectionSubsetSize, unsigned int);

  /** Get / Set and init the backprojection filter. The set function takes care
   * of initializing the mini-pipeline and the ramp filter must therefore be
   * created before calling this set function. */
  itkGetMacro(BackProjectionFilter, BackProjectionFilterPointer);
  virtual void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

protected:
  FDKConeBeamReconstructionFilter();
  ~FDKConeBeamReconstructionFilter() {}

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateData() ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  /** Pointers to each subfilter of this composite filter */
  typename ExtractFilterType::Pointer m_ExtractFilter;
  typename WeightFilterType::Pointer  m_WeightFilter;
  typename RampFilterType::Pointer    m_RampFilter;
  BackProjectionFilterPointer         m_BackProjectionFilter;

private:
  //purposely not implemented
  FDKConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

  /** Number of projections processed at a time. */
  unsigned int m_ProjectionSubsetSize;

  /** Probes to time reconstruction */
  itk::TimeProbe m_PreFilterProbe;
  itk::TimeProbe m_FilterProbe;
  itk::TimeProbe m_BackProjectionProbe;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFDKConeBeamReconstructionFilter.hxx"
#endif

#endif
