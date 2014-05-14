#ifndef __rtkSingleProjectionToFourDImageFilter_txx
#define __rtkSingleProjectionToFourDImageFilter_txx

#include "rtkSingleProjectionToFourDImageFilter.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

#include "rtkJosephBackProjectionImageFilter.h"

namespace rtk
{

template< typename VolumeSeriesType, typename VolumeType>
SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>::SingleProjectionToFourDImageFilter()
{
    this->SetNumberOfRequiredInputs(3);

    m_UseCuda=false; //Default behaviour is the CPU implementation

    // Create the two filters
//    m_SplatFilter = SplatFilterType::New();
    m_BackProjectionFilter = rtk::BackProjectionImageFilter<VolumeType, VolumeType>::New();
    m_BackProjectionFilter->ReleaseDataFlagOn();

//    // Connect them
//    m_SplatFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
//    m_SplatFilter->SetInputVolume(m_BackProjectionFilter->GetOutput());

}

template< typename VolumeSeriesType, typename VolumeType>
void SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
    this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename VolumeSeriesType, typename VolumeType>
void SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>::SetInputProjectionStack(const VolumeType* Projection)
{
    this->SetNthInput(1, const_cast<VolumeType*>(Projection));
}

template< typename VolumeSeriesType, typename VolumeType>
void SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>::SetInputEmptyVolume(const VolumeType* EmptyVolume)
{
    this->SetNthInput(2, const_cast<VolumeType*>(EmptyVolume));
}

template< typename VolumeSeriesType, typename VolumeType>
typename VolumeSeriesType::ConstPointer SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>::GetInputVolumeSeries()
{
    return static_cast< const VolumeSeriesType * >
            ( this->itk::ProcessObject::GetInput(0) );
}

template< typename VolumeSeriesType, typename VolumeType>
typename VolumeType::ConstPointer SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>::GetInputProjectionStack()
{
    return static_cast< const VolumeType * >
            ( this->itk::ProcessObject::GetInput(1) );
}

template< typename VolumeSeriesType, typename VolumeType>
typename VolumeType::ConstPointer SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>::GetInputEmptyVolume()
{
    return static_cast< const VolumeType * >
            ( this->itk::ProcessObject::GetInput(2) );
}

template< typename VolumeSeriesType, typename VolumeType>
void
SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>
::SetBackProjectionFilter (const BackProjectionFilterPointer _arg)
{
    // Configure and connect the back projection filter

    itkDebugMacro("setting BackProjectionFilter to " << _arg);
    if (this->m_BackProjectionFilter != _arg)
    {
        this->m_BackProjectionFilter = _arg;
        m_BackProjectionFilter->SetInput(0, this->GetInputEmptyVolume() );
        m_BackProjectionFilter->SetInput(1, this->GetInputProjectionStack() );
        m_BackProjectionFilter->SetTranspose(false);
        this->Modified();
    }
}


template< typename VolumeSeriesType, typename VolumeType>
void SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>::GenerateData()
{
    //The "input projection stack" is a single projection extracted from the full projection stack.
    //Its index is the number of the projection being used
    m_ProjectionNumber = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetIndex(2);


    // Create the splat filter
    if(m_UseCuda)
    {
//#if RTK_USE_CUDA
//        m_SplatFilter = rtk::CudaSplatImageFilter::New();
//#else
        std::cerr << "The program has not been compiled with cuda option" << std::endl;
//        return EXIT_FAILURE;
//#endif
    }
    else m_SplatFilter = SplatFilterType::New();
    m_SplatFilter->SetInputVolumeSeries(this->GetInputVolumeSeries());
    m_SplatFilter->SetInputVolume(m_BackProjectionFilter->GetOutput());
    m_SplatFilter->SetWeights(this->GetWeights());
    m_SplatFilter->SetProjectionNumber(m_ProjectionNumber);

    // We use the full geometry file, just like in SART, even though we only backproject one at a time
    // Which information is used in the geometry file depends on the index of the region extracted from the projection file
    m_BackProjectionFilter->SetGeometry(this->GetGeometry().GetPointer());
    m_BackProjectionFilter->GetOutput()->UpdateOutputInformation();
    m_BackProjectionFilter->GetOutput()->PropagateRequestedRegion();

    // Use a time probe
    itk::TimeProbe timeProbe;
    timeProbe.Start();
    m_BackProjectionFilter->Update();
    timeProbe.Stop();
    float backProjectionFilterExecutionTime = timeProbe.GetTotal();
//    std::cout << "Inside SingleProjectionToFourDImageFilter : Execution of backProjectionFilterExecutionTime took : " <<  backProjectionFilterExecutionTime << ' ' << timeProbe.GetUnit() << std::endl;

    timeProbe.Start();
    m_SplatFilter->Update();
    timeProbe.Stop();
    float SplatFilterExecutionTime = timeProbe.GetTotal() - backProjectionFilterExecutionTime;
//    std::cout << "Inside SingleProjectionToFourDImageFilter : Execution of SplatFilter took : " <<  SplatFilterExecutionTime << ' ' << timeProbe.GetUnit() << std::endl;


    this->GraftOutput( m_SplatFilter->GetOutput() );
}

}// end namespace


#endif
