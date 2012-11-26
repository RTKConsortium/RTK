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

#include "rtkDbf.h"

namespace rtk
{

DbfField::DbfField(std::string name, char type, unsigned char length, short recOffset) :
  m_Name(name),
  m_Type(type),
  m_Length(length),
  m_RecOffset(recOffset)
{
}

DbfFile::DbfFile(std::string fileName)
{
  m_Stream.open(fileName.c_str(), std::ios_base::in | std::ios_base::binary);
  if(!m_Stream.is_open() )
    return;

  // Ignore version and date
  m_Stream.seekg(4);

  // Read number of records, header size, record size
  m_Stream.read( (char*)&m_NumRecords, sizeof(m_NumRecords) );
  m_Stream.read( (char*)&m_HeaderSize, sizeof(m_HeaderSize) );
  m_Stream.read( (char*)&m_RecordSize, sizeof(m_RecordSize) );

  // Record allocation
  m_Record = new char[m_RecordSize];

  // The number of fields depends on the header size (32 are the blocks
  // describing the global part and each field, 1 is the terminator length)
  const unsigned int numFields = (m_HeaderSize - 32 - 1) / 32;

  // First byte is the deletion character. Then the field data are concatenated
  short fldRecOffset = 1;
  for(unsigned int i=0; i<numFields; i++) {

    // Go to beginning of current field structure description
    m_Stream.seekg(32 * (1 + i) );

    // Field names
    char fldName[11];
    m_Stream.read(fldName, 11);

    // Field types
    char fldType;
    m_Stream.read( (char*)&fldType, sizeof(fldType) );

    // Skip displacement of field in record?
    m_Stream.seekg(4, std::ios_base::cur);

    // Field length
    unsigned char fldLength;
    m_Stream.read( (char*)&fldLength, sizeof(fldLength) );

    // Add field and go to next
    m_Fields.push_back(DbfField(fldName, fldType, fldLength, fldRecOffset) );
    m_MapFieldNameIndex[m_Fields.back().GetName()] = i;

    fldRecOffset += fldLength;
    }

  // Seek to first record
  m_Stream.seekg(m_HeaderSize);
}

DbfFile::~DbfFile()
{
  delete [] m_Record;
}

bool DbfFile::ReadNextRecord()
{
  do
    {
    m_Stream.read(m_Record, m_RecordSize);
    }
  while(m_Stream.gcount() == m_RecordSize && m_Record[0] == 0x2A);
  return m_Stream.gcount() == m_RecordSize;
}

std::string DbfFile::GetFieldAsString(std::string fldName){
  DbfField &  field = m_Fields[ m_MapFieldNameIndex[fldName] ];
  std::string result( m_Record + field.GetRecOffset(), field.GetLength() );

  // Revome begin/end spaces
  std::string::size_type pos = result.find_last_not_of(' ');

  if(pos != std::string::npos) {
    result.erase(pos + 1);
    pos = result.find_first_not_of(' ');
    if(pos != std::string::npos)
      result.erase(0, pos);
    }
  else
    result.erase(result.begin(), result.end() );

  return result;
}

}
