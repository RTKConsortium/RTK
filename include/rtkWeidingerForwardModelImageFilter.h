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
#ifndef rtkWeidingerForwardModelImageFilter_h
#define rtkWeidingerForwardModelImageFilter_h

#include "itkImageToImageFilter.h"

//#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
//  #include <itkImageRegionSplitterDirection.h>
//#endif

#include "rtkMacro.h"

namespace rtk
{
  /** \class WeidingerForwardModelImageFilter
   * \brief Performs intermediate computations in Weidinger2016
   *
   * This filter performs all computations between forward and
   * back projection in Weidinger2016
   *
   * \author Cyril Mory
   *
   */
template< class TMaterialProjections,
          class TPhotonCounts,
          class TSpectrum,
          class TProjections=itk::Image<typename TMaterialProjections::PixelType::ValueType, TMaterialProjections::ImageDimension> >
class WeidingerForwardModelImageFilter : public itk::ImageToImageFilter<TMaterialProjections, TMaterialProjections>
{
public:
    /** Standard class typedefs. */
    typedef WeidingerForwardModelImageFilter                                    Self;
    typedef itk::ImageToImageFilter<TMaterialProjections, TMaterialProjections> Superclass;
    typedef itk::SmartPointer< Self >                                           Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(WeidingerForwardModelImageFilter, itk::ImageToImageFilter)

    /** Convenient parameters extracted from template types */
    itkStaticConstMacro(nBins, unsigned int, TPhotonCounts::PixelType::Dimension);
    itkStaticConstMacro(nMaterials, unsigned int, TMaterialProjections::PixelType::Dimension);
    itkStaticConstMacro(nEnergies, unsigned int, TSpectrum::PixelType::Dimension);

    /** Convenient typedef */
    typedef typename TMaterialProjections::PixelType::ValueType dataType;

    /** Define types for output images:
     * - one n-vector per pixel
     * - one nxn-matrix per pixel, but stored as one nxn-vector
    to allow vector back projection */
    typedef TMaterialProjections TOutputImage1;
    typedef itk::Vector<dataType, nMaterials * nMaterials> TPixelOutput2;
    typedef itk::Image<TPixelOutput2, TMaterialProjections::ImageDimension > TOutputImage2;

    /** Define the getters for the outputs, with correct types */
    TOutputImage1* GetOutput1();
    TOutputImage2* GetOutput2();

    /** Set methods for all inputs, since they have different types */
    void SetInputMaterialProjections(const TMaterialProjections* materialProjections);
    void SetInputPhotonCounts(const TPhotonCounts* photonCounts);
    void SetInputSpectrum(const TSpectrum* spectrum);
    void SetInputProjectionsOfOnes(const TProjections* projectionsOfOnes);

    /** Typedefs for additional input information */
    typedef itk::Matrix<dataType, nBins, nEnergies>       BinnedDetectorResponseType;
    typedef itk::Matrix<dataType, nEnergies, nMaterials>  MaterialAttenuationsType;

    /** Set and Get macros for the additional input information */
    itkGetConstReferenceMacro(BinnedDetectorResponse, BinnedDetectorResponseType)
    itkGetConstReferenceMacro(MaterialAttenuations, MaterialAttenuationsType)
    virtual void SetBinnedDetectorResponse(const BinnedDetectorResponseType & detResp);
    virtual void SetMaterialAttenuations(const MaterialAttenuationsType & matAtt);

protected:
    WeidingerForwardModelImageFilter();
    ~WeidingerForwardModelImageFilter() {}

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Does the real work. */
    void ThreadedGenerateData(const typename TOutputImage1::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;
    void BeforeThreadedGenerateData() ITK_OVERRIDE;

    /** Creates the Outputs */
    itk::DataObject::Pointer MakeOutput(unsigned int idx);

    /** Getters for the inputs */
    typename TMaterialProjections::ConstPointer GetInputMaterialProjections();
    typename TPhotonCounts::ConstPointer GetInputPhotonCounts();
    typename TSpectrum::ConstPointer GetInputSpectrum();
    typename TProjections::ConstPointer GetInputProjectionsOfOnes();

//#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
//    /** Splits the OutputRequestedRegion along the first direction, not the last */
//    const itk::ImageRegionSplitterBase* GetImageRegionSplitter(void) const ITK_OVERRIDE;
//    itk::ImageRegionSplitterDirection::Pointer  m_Splitter;
//#endif

    /** Additional input parameters */
    BinnedDetectorResponseType  m_BinnedDetectorResponse;
    MaterialAttenuationsType    m_MaterialAttenuations;

private:
    WeidingerForwardModelImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkWeidingerForwardModelImageFilter.hxx"
#endif

#endif
