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

// std include
#include <stdio.h>

#include "rtkXimImageIO.h"
#include <itkMetaDataObject.h>

template<typename T>
size_t rtk::XimImageIO::SetPropertyValue(char property_name[32], itk::uint32_t value_length, FILE *fp, Xim_header *xim)
{
	T property_value;
	T * unused_property_value;
	size_t addNelements = 0;

	if (value_length > 1)
	{
		unused_property_value = new T[value_length];
		addNelements += fread((void *)unused_property_value, sizeof(T), value_length, fp);
		return addNelements;
	}

	addNelements += fread((void *)&property_value, sizeof(T), value_length, fp);

	bool not_set = true;
	
	if (strncmp(property_name, "CouchLat", 8) == 0)
		xim->dCouchLat = property_value;
	else if (strncmp(property_name, "CouchLng", 8) == 0)
		xim->dCouchLng = property_value;
	else if (strncmp(property_name, "CouchVrt", 8) == 0)
		xim->dCouchVrt = property_value;
	else if (strncmp(property_name,"DataOffset", 10) == 0)
		xim->nPixelOffset = property_value;
	else if (strncmp(property_name, "KVSourceRtn", 9) == 0) //dGantryRtn
		xim->dCTProjectionAngle = property_value;
	else if (strncmp(property_name, "KVCollimatorX1", 14) == 0)
		xim->dCollX1 = property_value;
	else if (strncmp(property_name, "KVCollimatorX2", 14) == 0)
		xim->dCollX2 = property_value;
	else if (strncmp(property_name, "KVCollimatorY1", 14) == 0)
		xim->dCollY1 = property_value;
	else if (strncmp(property_name, "KVCollimatorY2", 14) == 0)
		xim->dCollY2 = property_value;
	else if (strncmp(property_name, "KVKiloVolts", 11) == 0)
		xim->dXRayKV = property_value;
	else if (strncmp(property_name, "KVMilliAmperes", 14) == 0)
		xim->dXRayMA = property_value;
	else if (strncmp(property_name, "KVNormChamber", 13) == 0)
		xim->dCTNormChamber = property_value;
	else if (strncmp(property_name, "KVSourceRtn", 11) == 0)
		xim->dCTSourceAngle = property_value; //May need off-set correction?
	else if (strncmp(property_name, "MMTrackingRemainderX", 20) == 0)
		xim->dGating4DInfoX = property_value;
	else if (strncmp(property_name, "MMTrackingRemainderY", 20) == 0)
		xim->dGating4DInfoY = property_value;
	else if (strncmp(property_name, "MMTrackingRemainderZ", 20) == 0)
		xim->dGating4DInfoZ = property_value;
	else if (strncmp(property_name, "MVCollimatorRtn", 15) == 0)
		xim->dCollRtn = property_value;
	else if (strncmp(property_name, "MVCollimatorX1", 14) == 0)
		xim->dCollX1 = property_value; // is this kV or MV??
	else if (strncmp(property_name, "MVCollimatorX2", 14) == 0)
		xim->dCollX2 = property_value;
	else if (strncmp(property_name, "MVCollimatorY1", 14) == 0)
		xim->dCollY1 = property_value;
	else if (strncmp(property_name, "MVCollimatorY2", 14) == 0)
		xim->dCollY2 = property_value;
	else if (strncmp(property_name, "MVDoseRate", 10) == 0)
		xim->dDoseRate = property_value;
	else if (strncmp(property_name, "MVEnergy", 8) == 0)
		xim->dEnergy = property_value;
	else if (strncmp(property_name, "PixelHeight", 11) == 0) //READ WRONGLY!
		xim->dIDUResolutionY = property_value * 10; 
	else if (strncmp(property_name, "PixelWidth", 10) == 0)
		xim->dIDUResolutionX = property_value * 10;
	else
		not_set = false;
	// Line below is for debugging:
	// std::cout << property_name << ":= " << property_value << (not_set?" Found":" Not Found") << std::endl;
	return addNelements;
}

//--------------------------------------------------------------------
// Read Image Information
void rtk::XimImageIO::ReadImageInformation()
{
  Xim_header xim;
  FILE *     fp;

  fp = fopen (m_FileName.c_str(), "rb");
  if (fp == NULL)
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);
  size_t nelements = 0;
  nelements += fread ( (void *) xim.sFileType, sizeof(char), 8, fp);
  nelements += fread ( (void *) &xim.FileVersion, sizeof(itk::int32_t), 1, fp);
  nelements += fread ( (void *) &xim.SizeX, sizeof(itk::int32_t), 1, fp);
  nelements += fread ( (void *) &xim.SizeY, sizeof(itk::int32_t), 1, fp);

  nelements += fread((void *)&xim.dBitsPerPixel, sizeof(itk::int32_t), 1, fp);
  nelements += fread((void *)&xim.dBytesPerPixel, sizeof(itk::int32_t), 1, fp);
  m_bytes_per_pixel = xim.dBytesPerPixel;
  nelements += fread((void *)&xim.dCompressionIndicator, sizeof(itk::int32_t), 1, fp);
  imageDataStart = ftell(fp);
  if (xim.dCompressionIndicator == 1) // True for Scripps Data
  {
	  nelements += fread((void *)&xim.lookUpTableSize, sizeof(itk::int32_t), 1, fp);
	  fseek(fp, xim.lookUpTableSize, SEEK_CUR);
	  nelements += fread((void *)&xim.compressedPixelBufferSize, sizeof(itk::int32_t), 1, fp);
	  fseek(fp, xim.compressedPixelBufferSize, SEEK_CUR);
	  nelements += fread((void *)&xim.unCompressedPixelBufferSize, sizeof(itk::int32_t), 1, fp);
	  if (nelements != /*char*/8 +/*itk::int32_t*/9) // + /*itk::int8*/(xim.lookUpTableSize * 4))
		  std::cout << "Could not read header data in " << m_FileName; //itkGenericExceptionMacro();
  }
  else
  {
	  nelements += fread((void *)&xim.unCompressedPixelBufferSize, sizeof(itk::int32_t), 1, fp);
	  fseek(fp, xim.unCompressedPixelBufferSize, SEEK_CUR);
	  if (nelements != /*char*/8 +/*itk::uint32_t*/7) 
		  itkGenericExceptionMacro(<< "Could not read header data in " << m_FileName);
  }
  
  // Histogram Reading:
  size_t nhistElements = 0;
  nhistElements += fread((void *)&xim.binsInHistogram, sizeof(itk::int32_t), 1, fp);
  
  xim.histogramData = new int[xim.binsInHistogram];
  for (itk::int32_t i = 0; i < xim.binsInHistogram; i++)
	xim.histogramData[i] = 0;

  nhistElements += fread((void *)xim.histogramData, sizeof(itk::int32_t), xim.binsInHistogram, fp);
  if (nhistElements != (xim.binsInHistogram + 1))
  {
	  itkGenericExceptionMacro(<< "Could not read histogram from header data in " << m_FileName);
  }
  // Properties Readding:
  nelements += fread((void *)&xim.numberOfProperties, sizeof(itk::int32_t), 1, fp);
  itk::int32_t property_name_length;
  char property_name[32];
  itk::int32_t property_type;
  itk::int32_t property_value_length = 0;
  size_t theoretical_nelements = nelements; // Same as reseting
  
  for (size_t i = 0; i < xim.numberOfProperties; i++)
  {
	  nelements += fread((void *)&property_name_length, sizeof(itk::int32_t), 1, fp);
	  nelements += fread((void *)&property_name, sizeof(char), property_name_length, fp);
	  
	  // std::cout << "Property name: " << property_name << ", "; //FOR DEBUGGING ONLY REMOVE WHEN IT WORKS!!
	  nelements += fread((void *)&property_type, sizeof(itk::int32_t), 1, fp);
	  theoretical_nelements += property_name_length + 2;
	  // std::cout << property_type << ", ";
	  switch (property_type)
	  {
	  case 0://property_value type = uint32
		  nelements += SetPropertyValue<itk::int32_t>(property_name, 1, fp, &xim);
		  theoretical_nelements++;
		  break;
	  case 1://property_value type = double
		  theoretical_nelements++;
		  nelements += SetPropertyValue<double>(property_name, 1, fp, &xim);
		  break;
	  case 2://property_value type = length * char
	      nelements += fread((void *)&property_value_length, sizeof(itk::int32_t), 1, fp);
		  theoretical_nelements += property_value_length+1;
		  nelements += SetPropertyValue<char>(property_name, property_value_length, fp, &xim);
		  break;
	  case 4://property_value type = length * double
		  nelements += fread((void *)&property_value_length, sizeof(itk::int32_t), 1, fp);
		  nelements += SetPropertyValue<double>(property_name, property_value_length/8, fp, &xim);
		  theoretical_nelements += property_value_length/8+1;
		  break;
	  case 5://property_value type = length * uint32
		  nelements += fread((void *)&property_value_length, sizeof(itk::int32_t), 1, fp);
		  nelements += SetPropertyValue<itk::int32_t>(property_name, property_value_length/4, fp, &xim);
		  theoretical_nelements += property_value_length/4+1;
		  break;
	  default:
		  std::cout << "\n\nProperty name: " << property_name << ", type: " << property_type << ", is not supported! ABORTING decoding!";
		  return;
	  }
  }
  if (nelements != theoretical_nelements){
	  std::cout << nelements << " != " << theoretical_nelements << std::endl;
	  itkGenericExceptionMacro(<< "Could not read properties of " << m_FileName);
  }
	
  
  if(fclose (fp) != 0)
    itkGenericExceptionMacro(<< "Could not close file: " << m_FileName);

  /* Convert xim to ITK image information */
  SetNumberOfDimensions(2);
  SetDimensions(0, xim.SizeX);
  SetDimensions(1, xim.SizeY);

  SetSpacing(0, xim.dIDUResolutionX); // set to PixelHeight/Width
  SetSpacing(1, xim.dIDUResolutionY);
  SetOrigin(0, -0.5 * (xim.SizeX - 1) * xim.dIDUResolutionX); //SR: assumed centered
  SetOrigin(1, -0.5 * (xim.SizeY - 1) * xim.dIDUResolutionY); //SR: assumed centered

  SetComponentType(itk::ImageIOBase::UINT);
  /* Store important meta information in the meta data dictionary */
  if (xim.SizeX * xim.SizeY != 0)
	itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTProjectionAngle", xim.dCTProjectionAngle);
  else
	itk::EncapsulateMetaData<double>(this->GetMetaDataDictionary(), "dCTProjectionAngle", 6000);
}
//--------------------------------------------------------------------
bool rtk::XimImageIO::CanReadFile(const char* FileNameToRead)
{
	std::string                  filename(FileNameToRead);
	const std::string::size_type it = filename.find_last_of(".");
	std::string                  fileExt(filename, it + 1, filename.length());

	if (fileExt != std::string("xim"))
		return false;
	return true;
}

//--------------------------------------------------------------------
// Read Image Content
void rtk::XimImageIO::Read(void * buffer)
{
  
  FILE *fp;
  itk::uint32_t *buf = (itk::uint32_t*)buffer;
  itk::int32_t  a;
  int i;

  fp = fopen (m_FileName.c_str(), "rb");
  if (fp == NULL)
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);

  if(fseek (fp, imageDataStart, SEEK_SET) != 0) 
    itkGenericExceptionMacro(<< "Could not seek to image data in: " << m_FileName);

  size_t nelements = 0;
  itk::int32_t lookUpTableSize;
  itk::int32_t compressedPixelBufferSize;
  // De"compress" image
  nelements += fread((void *)&lookUpTableSize, sizeof(itk::int32_t), 1, fp);
  unsigned _int8 * m_lookup_table = (unsigned _int8*) malloc(sizeof(unsigned _int8) * lookUpTableSize);
  nelements += fread((void *)m_lookup_table, sizeof(unsigned _int8), lookUpTableSize, fp);
  
  nelements += fread((void *)&compressedPixelBufferSize, sizeof(itk::int32_t), 1, fp);

  for (i = 0; i < (GetDimensions(0) + 1); i++) {
	if (1 != fread(&a, sizeof(itk::uint32_t), 1, fp))
	  itkGenericExceptionMacro(<< "Could not read first row +1 in: " << m_FileName);
	buf[i] = a;
  }
  
  int lookup_table_pos = 0;
  char  diff8;
  short diff16;
  long  diff32, diff = 0;
  int lut_off = 0, lut_idx = 0;
  unsigned char v;

  while( i < (GetDimensions(0)*GetDimensions(1))){
	  v = m_lookup_table[lut_idx];
	  switch (lut_off) {
	  case 0:
		  v = v & 0x03;
		  lut_off++;
		  break;
	  case 1:
		  v = (v & 0x0C) >> 2;
		  lut_off++;
		  break;
	  case 2:
		  v = (v & 0x30) >> 4;
		  lut_off++;
		  break;
	  case 3:
		  v = (v & 0xC0) >> 6;
		  lut_off = 0;
		  lut_idx++;
		  break;
	  }
	  switch (v) {
	  case 0:
		  nelements += fread(&diff8, sizeof(unsigned char), 1, fp);
		  diff = diff8;
		  break;
	  case 1:
		  nelements += fread(&diff16, sizeof(unsigned short), 1, fp);
		  diff = diff16;
		  break;
	  case 2:
		  nelements += fread(&diff32, sizeof(itk::uint32_t), 1, fp);
		  diff = diff32;
		  break;
	  }
	buf[i] = diff + buf[i - 1] + buf[i - GetDimensions(0)] - buf[i - GetDimensions(0) - 1];
	i++;
  }

  /* Clean up */
  free(m_lookup_table);
  if(fclose (fp) != 0)
    itkGenericExceptionMacro(<< "Could not close file: " << m_FileName);
  return;

}

//--------------------------------------------------------------------
bool rtk::XimImageIO::CanWriteFile(const char* itkNotUsed(FileNameToWrite))
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void rtk::XimImageIO::Write(const void* itkNotUsed(buffer))
{
  //TODO?
}
