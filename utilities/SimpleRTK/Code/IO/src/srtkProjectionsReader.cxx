/*=========================================================================
 *
 *  Copyright Insight Software Consortium & RTK Consortium
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
#ifdef _MFC_VER
#pragma warning(disable:4996)
#endif

#include "srtkProjectionsReader.h"

#include <rtkConfiguration.h>
#ifdef RTK_USE_CUDA
# include <itkCudaImage.h>
#endif
#include <rtkProjectionsReader.h>
#include <itkChangeInformationImageFilter.h>

namespace rtk {
  namespace simple {

  Image ReadProjections ( const std::vector<std::string> &filenames )
    {
    ProjectionsReader reader;
    return reader.SetFileNames ( filenames ).Execute();
    }


  ProjectionsReader::ProjectionsReader()
    {

    // list of pixel types supported
    typedef RealPixelIDTypeList PixelIDTypeList;

    this->m_MemberFactory.reset( new detail::MemberFunctionFactory<MemberFunctionType>( this ) );

    this->m_MemberFactory->RegisterMemberFunctions< PixelIDTypeList, 3 > ();
    this->m_MemberFactory->RegisterMemberFunctions< PixelIDTypeList, 2 > ();
    }

  std::string ProjectionsReader::ToString() const {

      std::ostringstream out;
      out << "rtk::simple::ProjectionsReader";
      out << std::endl;

      out << "  FileNames: " << std::endl;
      std::vector<std::string>::const_iterator iter  = m_FileNames.begin();
      while( iter != m_FileNames.end() )
        {
        std::cout << "    \"" << *iter << "\"" << std::endl;
        ++iter;
        }

      return out.str();
    }

  ProjectionsReader& ProjectionsReader::SetFileNames ( const std::vector<std::string> &filenames )
    {
    this->m_FileNames = filenames;
    return *this;
    }

  const std::vector<std::string> &ProjectionsReader::GetFileNames() const
    {
    return this->m_FileNames;
    }

  Image ProjectionsReader::Execute ()
    {
    // todo check if filename does not exits for robust error handling
    assert( !this->m_FileNames.empty() );

    PixelIDValueType type = rtk::simple::srtkFloat32;
    unsigned int dimension = 3;

    return this->m_MemberFactory->GetMemberFunction( type, dimension )();
    }

  template <class TImageType> Image
  ProjectionsReader::ExecuteInternal( void )
    {

    typedef TImageType                        ImageType;
    typedef rtk::ProjectionsReader<ImageType> Reader;

    // if the IsInstantiated is correctly implemented this should
    // not occour
    assert( ImageTypeToPixelIDValue<ImageType>::Result != (int)srtkUnknown );
    typename Reader::Pointer reader = Reader::New();
    reader->SetFileNames( this->m_FileNames );

    this->PreUpdate( reader.GetPointer() );

    reader->Update();

    // We must change the output to ensure that there is a 0 index
    typename ImageType::RegionType lpr = reader->GetOutput()->GetLargestPossibleRegion();
    typename ImageType::RegionType::IndexType idx = lpr.GetIndex();
    typename ImageType::PointType newOrig;
    reader->GetOutput()->TransformIndexToPhysicalPoint(idx, newOrig);
    typename ImageType::RegionType::OffsetType offset;
    for(unsigned int i=0; i<ImageType::GetImageDimension(); i++)
      offset[i] = idx[i] * -1;

    typedef itk::ChangeInformationImageFilter<ImageType> ChangeInformationType;
    typename ChangeInformationType::Pointer changeInfo = ChangeInformationType::New();
    changeInfo->SetInput( reader->GetOutput() );
    changeInfo->ChangeOriginOn();
    changeInfo->SetOutputOrigin(newOrig);
    changeInfo->ChangeRegionOn();
    changeInfo->SetOutputOffset(offset);
    changeInfo->Update();

    return Image( changeInfo->GetOutput() );
    }
  }
}
