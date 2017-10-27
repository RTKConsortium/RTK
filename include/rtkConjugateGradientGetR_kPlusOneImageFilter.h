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

#ifndef rtkConjugateGradientGetR_kPlusOneImageFilter_h
#define rtkConjugateGradientGetR_kPlusOneImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkBarrier.h>

#include "rtkConfiguration.h"
#include "rtkMacro.h"

namespace rtk
{
template< typename TInputImage>
class ConjugateGradientGetR_kPlusOneImageFilter : public itk::ImageToImageFilter< TInputImage, TInputImage>
{
public:
    /** Standard class typedefs. */
    typedef ConjugateGradientGetR_kPlusOneImageFilter          Self;
    typedef itk::ImageToImageFilter< TInputImage, TInputImage> Superclass;
    typedef itk::SmartPointer< Self >                          Pointer;
    typedef typename TInputImage::RegionType                   OutputImageRegionType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ConjugateGradientGetR_kPlusOneImageFilter, itk::ImageToImageFilter)

    /** Functions to set the inputs */
    void SetRk(const TInputImage* Rk);
    void SetPk(const TInputImage* Pk);
    void SetAPk(const TInputImage* APk);

    itkGetMacro(Alphak, float)
    itkGetMacro(SquaredNormR_k, float)
    itkGetMacro(SquaredNormR_kPlusOne, float)

protected:
    ConjugateGradientGetR_kPlusOneImageFilter();
    ~ConjugateGradientGetR_kPlusOneImageFilter() {}

    typename TInputImage::Pointer GetRk();
    typename TInputImage::Pointer GetPk();
    typename TInputImage::Pointer GetAPk();

    /** Initialize the thread synchronization barrier before the threads run,
        and create a few vectors in which each thread will store temporary
        accumulation results */
    void BeforeThreadedGenerateData() ITK_OVERRIDE;

    /** Do the real work */
    void ThreadedGenerateData(const typename TInputImage::RegionType &
                               outputRegionForThread,
                               ThreadIdType threadId) ITK_OVERRIDE;

    /**  Set m_alphak to its correct value as it has to be passed to other filters */
    void AfterThreadedGenerateData() ITK_OVERRIDE;

private:
    ConjugateGradientGetR_kPlusOneImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
    float m_Alphak;
    float m_SquaredNormR_k;
    float m_SquaredNormR_kPlusOne;

    // Thread synchronization tool
    itk::Barrier::Pointer m_Barrier;

    // These vector store one accumulation value per thread
    // The values are then sumed
    std::vector<float> m_SquaredNormR_kVector;
    std::vector<float> m_SquaredNormR_kPlusOneVector;
    std::vector<float> m_PktApkVector;

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientGetR_kPlusOneImageFilter.hxx"
#endif

#endif
