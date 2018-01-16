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

#ifndef rtkBioscanGeometryReader_hxx
#define rtkBioscanGeometryReader_hxx

#include "rtkMacro.h"
#include "rtkBioscanGeometryReader.h"
#include "rtkIOFactories.h"

#include <gdcmImageReader.h>
#include <gdcmAttribute.h>
#include <gdcmDataSet.h>

namespace rtk
{

std::vector<float>
BioscanGeometryReader::
GetVectorTagValue(const gdcm::DataSet & ds, uint16_t group, uint16_t element) const
{
  const gdcm::Tag tag(group, element);
  if( !ds.FindDataElement( tag ) )
    {
    itkExceptionMacro(<< "Cannot find tag " << group << "|" << element);
    }
  const gdcm::DataElement & de = ds.GetDataElement(tag);
  gdcm::Element<gdcm::VR::FL,gdcm::VM::VM1_n> el;
  el.Set( de.GetValue() );
  std::vector<float> val;
  for(int i=0; i<el.GetLength(); i++)
    {
    val.push_back(el.GetValue(i));
    }
  return val;
}

std::string
BioscanGeometryReader::
GetStringTagValue(const gdcm::DataSet & ds, uint16_t group, uint16_t element) const
{
  const gdcm::Tag tag(group, element);
  if( !ds.FindDataElement( tag ) )
    {
    itkExceptionMacro(<< "Cannot find tag " << group << "|" << element);
    }
  const gdcm::DataElement & de = ds.GetDataElement(tag);
  const gdcm::ByteValue * bv = de.GetByteValue();
  return std::string( bv->GetPointer(), bv->GetLength() );
}

double
BioscanGeometryReader::
GetFloatTagValue(const gdcm::DataSet & ds, uint16_t group, uint16_t element) const
{
  const gdcm::Tag tag(group, element);
  if( !ds.FindDataElement( tag ) )
    {
    itkExceptionMacro(<< "Cannot find tag " << group << "|" << element);
    }
  const gdcm::DataElement & de = ds.GetDataElement(tag);
  gdcm::Element<gdcm::VR::FD,gdcm::VM::VM1> el;
  el.SetFromDataElement( de );
  return el.GetValue();
}

void
BioscanGeometryReader::
GenerateData()
{
  m_Geometry = GeometryType::New();
  for(size_t noProj=0; noProj < m_ProjectionsFileNames.size(); noProj++)
    {
    gdcm::Reader reader;
    reader.SetFileName( m_ProjectionsFileNames[noProj].c_str() );
    if ( !reader.Read() )
      {
      itkExceptionMacro(<< "Cannot read requested file");
      }
    const gdcm::DataSet & ds =  reader.GetFile().GetDataSet();

    // See https://github.com/JStrydhorst/win-cone-ct/blob/master/ct_recon_win.h#L111
    const std::vector<float> zOffsets = GetVectorTagValue(ds, 0x0009, 0x1046);
    const std::vector<float> yOffsets = GetVectorTagValue(ds, 0x0009, 0x1047);
    const double sdd = atof(GetStringTagValue(ds, 0x0018, 0x1110).c_str());
    const double sid = atof(GetStringTagValue(ds, 0x0018, 0x1111).c_str());
    //const double spacing = GetFloatTagValue(ds, 0x0018, 0x9306);
    const double angle = GetFloatTagValue(ds, 0x0009, 0x1036);

    // See https://github.com/JStrydhorst/win-cone-ct/blob/master/ct_recon_win.h#L222
    const float yOffset = yOffsets[(itk::Math::Round<int>(angle)+180)%360];
    const float zOffset = zOffsets[(itk::Math::Round<int>(angle)+180)%360];
    if(std::string("BLANK SCAN") != GetStringTagValue(ds, 0x0008, 0x0008))
      m_Geometry->AddProjection(sid,
                                sdd,
                                angle,
                                yOffset,
                                zOffset);
    }
}
} //namespace rtk
#endif
