/*=========================================================================
*
*  Copyright Insight Software Consortium
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

// This one header will include all SimpleITK filters and external
// objects.
#include <SimpleRTK.h>
#include <iostream>
#include <stdlib.h>

// create convenient namespace alias
namespace srtk = rtk::simple;

int main ( int argc, char* argv[] ) {

  if ( argc < 2 ) {
    std::cerr << "Usage: " << argv[0] << " <output>\n";
    return 1;
  }

  // Defines the RTK geometry object
  srtk::ThreeDimCircularProjectionGeometry geometry;
  unsigned int numberOfProjections = 360;
  float firstAngle = 0;
  float angularArc = 360;
  float sid = 600; // source to isocenter distance in mm
  float sdd = 1200; // source to detector distance in mm
  float isox = 0; // X coordinate on the projection image of isocenter
  float isoy = 0; // Y coordinate on the projection image of isocenter
  for (unsigned int x = 0; x < numberOfProjections; x++)
    {
    float angle = firstAngle + x * angularArc / numberOfProjections;
    geometry.AddProjection(sid, sdd, angle, isox, isoy);
    }
  srtk::ConstantImageSource constantImageSource;
  std::vector<double> origin(3,-127.0);
  std::vector<unsigned int> sizeOutput(3);
  sizeOutput[0]=256;
  sizeOutput[1]=256;
  sizeOutput[2]=numberOfProjections;
  std::vector<double> spacing(3,1.0);
  constantImageSource.SetOrigin(origin);
  constantImageSource.SetSpacing(spacing);
  constantImageSource.SetSize(sizeOutput);
  constantImageSource.SetConstant(0.0);
  srtk::Image source = constantImageSource.Execute();
  
  srtk::RayEllipsoidIntersectionImageFilter rei;
  std::vector<double> semiprincipalaxis(3,50.0);
  std::vector<double> center(3,0.0);
  // Set GrayScale value, axes, center...
  rei.SetDensity(20);
  rei.SetAngle(0);
  rei.SetCenter(center);
  rei.SetAxis(semiprincipalaxis);
  rei.SetGeometry( &geometry );
  srtk::Image reiImage = rei.Execute(source);
  
  // Create reconstructed image
  srtk::ConstantImageSource constantImageSource2;
  std::vector<unsigned int> sizeOutput2(3,256);
  constantImageSource2.SetOrigin( origin );
  constantImageSource2.SetSpacing( spacing );
  constantImageSource2.SetSize( sizeOutput2 );
  constantImageSource2.SetConstant(0.0);
  srtk::Image source2 = constantImageSource2.Execute();

  std::cout << "Performing reconstruction" << std::endl;
  srtk::FDKConeBeamReconstructionFilter feldkamp;
  feldkamp.SetGeometry( &geometry );
  feldkamp.SetTruncationCorrection(0.0);
  feldkamp.SetHannCutFrequency(0.0);
  srtk::Image image = feldkamp.Execute(source2,reiImage);

  srtk::ImageFileWriter writer;
  writer.SetFileName(argv[1]);
  writer.Execute(image);
  return 0;
}
