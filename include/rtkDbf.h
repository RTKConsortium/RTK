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

#ifndef rtkDbf_h
#define rtkDbf_h

#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <stdlib.h>

namespace rtk
{

/** \class DbfField
 * 
 * Class for the description of a dbase field.
 */
class DbfField
{
public:
  /** Constructor */
  DbfField(std::string name, char type, unsigned char length, short recOffset);

  /** Basic field properties stored in the header of the dbf file */
  std::string GetName()           {return m_Name;}
  char        GetType()           {return m_Type;}
  short       GetLength()         {return m_Length;}

  /** Memory offset from beginning of the record */
  short       GetRecOffset()      {return m_RecOffset;}

private:
  std::string m_Name;
  char        m_Type;
  short       m_Length;
  short       m_RecOffset;
};

/** \class DbfFile
 * 
 * Light dbase file (.dbf) file reader. It assumes little-endianness
 * (least significant byte first). The format describet on this page:
 * http://www.dbf2002.com/dbf-file-format.html
 */
class DbfFile
{
public:
  /** Constructor initializes the structure and goes to first record */
  DbfFile(std::string fileName);
  ~DbfFile();

  /** Return open status of file stream */
  bool is_open() { return m_Stream.is_open(); }

  /** Number of records contained in the tabe */
  size_t GetNumberOfRecords()      { return m_Fields.size(); }

  /** Read in memory the next record. Return true if successful and false
    oftherwise. */
  bool ReadNextRecord();

  /** Access to field value of field named fldName */
  std::string GetFieldAsString(std::string fldName);

  double GetFieldAsDouble(std::string fldName) { return atof(GetFieldAsString(fldName).c_str() ); }

private:
  /** File stream. AFter constructor, positionned to next record to read. */
  std::ifstream m_Stream;

  /** Global properties of a dbf file */
  unsigned int m_NumRecords;
  unsigned short m_RecordSize;
  unsigned short m_HeaderSize;

  /** Set of fields described in the header */
  std::vector<DbfField> m_Fields;

  /** Map between field names and field index */
  std::map<std::string, unsigned int> m_MapFieldNameIndex;

  /** Current record in memory */
  char *m_Record;
};
}

#endif
