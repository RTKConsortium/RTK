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

#include "rtki0estimation_ggo.h"
#include "rtkMacro.h"
#include "rtkGgoFunctions.h"

#include <itkExtractImageFilter.h>
#include "rtkI0EstimationProjectionFilter.h"
#include "rtkProjectionsReader.h"

#include <vector>

int main(int argc, char * argv[])
{
	GGO(rtki0estimation, args_info);

	typedef unsigned short OutputPixelType;
	const unsigned int Dimension = 3;
	typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

	typedef itk::RegularExpressionSeriesFileNames RegexpType;
	RegexpType::Pointer names = RegexpType::New();
	names->SetDirectory(args_info.path_arg);
	names->SetNumericSort(args_info.nsort_flag);
	names->SetRegularExpression(args_info.regexp_arg);

	typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileNames(names->GetFileNames());
	reader->UpdateOutputInformation();
	
	typedef itk::ExtractImageFilter<OutputImageType, OutputImageType> ExtractFilterType;
	ExtractFilterType::Pointer extract = ExtractFilterType::New();
	extract->InPlaceOff();
	extract->SetDirectionCollapseToSubmatrix();
	extract->SetInput(reader->GetOutput());
	
	ExtractFilterType::InputImageRegionType subsetRegion = reader->GetOutput()->GetLargestPossibleRegion();
	subsetRegion = reader->GetOutput()->GetLargestPossibleRegion();
	extract->SetInput(reader->GetOutput());
	OutputImageType::SizeType extractSize = subsetRegion.GetSize();
	extractSize[2] = 1;
	OutputImageType::IndexType start = subsetRegion.GetIndex();

	typedef rtk::I0EstimationProjectionFilter<1> I0FilterType;

	std::vector<unsigned short> I0buffer;

	for (unsigned int i = 0; i < (int)subsetRegion.GetSize()[2]; i++)
	{	
		I0FilterType::Pointer i0est = I0FilterType::New();
		i0est->SetInput(extract->GetOutput());
		
		start[2] = i;
		OutputImageType::RegionType desiredRegion(start, extractSize);
		extract->SetExtractionRegion(desiredRegion);
				
		try {
			i0est->Update();
			I0buffer.push_back(i0est->GetI0());
		}
		catch (itk::ExceptionObject & err) {
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
			return EXIT_FAILURE;
		}
	}
	
	ofstream paramFile;
	paramFile.open(args_info.output_arg);
	std::vector<unsigned short>::const_iterator it = I0buffer.begin();
	for (; it != I0buffer.end(); ++it) {
		paramFile << *it << ",";
	}
	paramFile.close();
	
	return EXIT_SUCCESS;
}
