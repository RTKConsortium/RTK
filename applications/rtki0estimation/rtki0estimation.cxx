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
#include <itkImageFileWriter.h>
#include "rtkI0EstimationProjectionFilter.h"
#include "rtkProjectionsReader.h"

#include <vector>
#include <algorithm>
#include <string>

int main(int argc, char * argv[])
{
	GGO(rtki0estimation, args_info);

	typedef unsigned short InputPixelType;
	const unsigned int Dimension = 3;
	typedef itk::Image< InputPixelType, Dimension > InputImageType;
	typedef itk::Image< unsigned, Dimension > OutputHistogramType;

	typedef itk::RegularExpressionSeriesFileNames RegexpType;
	RegexpType::Pointer names = RegexpType::New();
	names->SetDirectory(args_info.path_arg);
	names->SetNumericSort(args_info.nsort_flag);
	names->SetRegularExpression(args_info.regexp_arg);

	typedef rtk::ProjectionsReader< InputImageType > ReaderType;
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileNames(names->GetFileNames());
	reader->UpdateOutputInformation();
		
	typedef itk::ExtractImageFilter<InputImageType, InputImageType> ExtractFilterType;
	ExtractFilterType::Pointer extract = ExtractFilterType::New();
	extract->InPlaceOff();
	extract->SetDirectionCollapseToSubmatrix();
	extract->SetInput(reader->GetOutput());
	
	ExtractFilterType::InputImageRegionType subsetRegion = reader->GetOutput()->GetLargestPossibleRegion();
	subsetRegion = reader->GetOutput()->GetLargestPossibleRegion();
	extract->SetInput(reader->GetOutput());
	InputImageType::SizeType extractSize = subsetRegion.GetSize();
	extractSize[2] = 1;
	InputImageType::IndexType start = subsetRegion.GetIndex();

	typedef rtk::I0EstimationProjectionFilter<2> I0FilterType;

	std::vector<unsigned short> I0buffer;
	std::vector<float> Variables;

	std::vector<int> range_mod(3, 0.);
	for (unsigned int i = 0; i < 3; i++)
		range_mod[i] = args_info.range_arg[i];

	unsigned int imin = 0;
	unsigned int istep = 1;
	unsigned int imax = (unsigned int)subsetRegion.GetSize()[2];
	if ((args_info.range_arg[0] <= args_info.range_arg[2]) && (istep <= (args_info.range_arg[2] - args_info.range_arg[0]))) {
		imin = args_info.range_arg[0];
		istep = args_info.range_arg[1];
		imax = std::min(unsigned(args_info.range_arg[2]), imax);
	} 
	std::cout << imin << " " << imax << " " << istep << std::endl;
	
	OutputHistogramType::Pointer image;

	I0FilterType::Pointer i0est = I0FilterType::New();
	for (unsigned int i = imin; i < imax; i+=istep)
	{
		std::cout << "Image " << i << std::endl;
	
		if (args_info.expected_arg != 65535){
			i0est->SetExpectedI0(args_info.expected_arg);
		}
		if (args_info.median_flag) {
			i0est->MedianOn();
		}
		if (args_info.rls_flag){
			i0est->UseRLSOn();
		}
		if (args_info.turbo_flag) {
			i0est->UseTurboOn();
		}
		i0est->SetInput(extract->GetOutput());
		
		start[2] = i;
		InputImageType::RegionType desiredRegion(start, extractSize);
		extract->SetExtractionRegion(desiredRegion);
				
		try {
			i0est->UpdateLargestPossibleRegion();
		}
		catch (itk::ExceptionObject & err) {
			std::cerr << "ExceptionObject caught !" << std::endl;
			std::cerr << err << std::endl;
			return EXIT_FAILURE;
		}

		image = i0est->GetOutput();

		I0buffer.push_back(i0est->GetI0());
		I0buffer.push_back(i0est->GetI0mean());
		I0buffer.push_back(i0est->GetImin());
		I0buffer.push_back(i0est->GetImax());
		I0buffer.push_back(i0est->GetIrange());
		I0buffer.push_back(i0est->GetlowBound());
		I0buffer.push_back(i0est->GethighBound());
		I0buffer.push_back(i0est->GetlowBndRls());
		I0buffer.push_back(i0est->GethighBndRls());
		I0buffer.push_back(i0est->GetI0fwhm());
		Variables.push_back(i0est->GetI0rls());
		Variables.push_back(i0est->GetI0sigma());
	}
	
	ofstream paramFile;
	paramFile.open(args_info.debug_arg);
	std::vector<unsigned short>::const_iterator it = I0buffer.begin();
	for (; it != I0buffer.end(); ++it) {
		paramFile << *it << ",";
	}
	paramFile.close();

	char strvar[100] = "var_";
	paramFile.open(std::strcat(strvar, args_info.debug_arg));
	std::vector<float>::const_iterator itf = Variables.begin();
	for (; itf != Variables.end(); ++itf) {
		paramFile << *itf << ",";
	}
	paramFile.close();
	
	typedef itk::ImageFileWriter<OutputHistogramType> ImageWriter;
	ImageWriter::Pointer writer = ImageWriter::New();
	writer->SetFileName("Output.mhd");
	writer->SetInput(image);
	try {
		writer->Update();
	}
	catch (itk::ExceptionObject & err) {
		std::cerr << "ExceptionObject caught !" << std::endl;
		std::cerr << err << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
