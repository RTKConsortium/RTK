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
using System;
using rtk.simple;

namespace rtk.simple.examples {
    class RTKFirstReconstruction {
        static void Main(string[] args) {
            try {
                if (args.Length < 1) {
                    Console.WriteLine("Usage: RTKFirstReconstruction <output>");
                    return;
                }

                // Defines the RTK geometry object
                ThreeDimCircularProjectionGeometry geometry = new ThreeDimCircularProjectionGeometry();
                uint numberOfProjections = 360;
                float firstAngle = 0;
                float angularArc = 360;
                float sid = 600; // source to isocenter distance in mm
                float sdd = 1200; // source to detector distance in mm
                float isox = 0; // X coordinate on the projection image of isocenter
                float isoy = 0; // Y coordinate on the projection image of isocenter
                for (int x = 0; x < numberOfProjections; x++)
                  {
                  float angle = firstAngle + x * angularArc / numberOfProjections;
                  geometry.AddProjection(sid, sdd, angle, isox, isoy);
                  }
                ConstantImageSource constantImageSource = new ConstantImageSource();
                VectorDouble origin = new VectorDouble(3);
                origin.Add(-127.0);
                origin.Add(-127.0);
                origin.Add(-127.0);
                VectorUInt32 sizeOutput = new VectorUInt32(3);
                sizeOutput.Add(256);
                sizeOutput.Add(256);
                sizeOutput.Add(numberOfProjections);
                VectorDouble spacing = new VectorDouble(3);
                spacing.Add(1.0);
                spacing.Add(1.0);
                spacing.Add(1.0);

                constantImageSource.SetOrigin(origin);
                constantImageSource.SetSpacing(spacing);
                constantImageSource.SetSize(sizeOutput);
                constantImageSource.SetConstant(0.0);
                Image source = constantImageSource.Execute();

                RayEllipsoidIntersectionImageFilter rei = new RayEllipsoidIntersectionImageFilter();
                VectorDouble semiprincipalaxis = new VectorDouble(3);
                semiprincipalaxis.Add(50.0);
                semiprincipalaxis.Add(50.0);
                semiprincipalaxis.Add(50.0);

                VectorDouble center = new VectorDouble(3);
                center.Add(0.0);
                center.Add(0.0);
                center.Add(0.0);
                // Set GrayScale value, axes, center...
                rei.SetDensity(20);
                rei.SetAngle(0);
                rei.SetCenter(center);
                rei.SetAxis(semiprincipalaxis);
                rei.SetGeometry(geometry);
                Image reiImage = rei.Execute(source);


                // Create reconstructed image
                ConstantImageSource constantImageSource2 = new ConstantImageSource();
                VectorUInt32 sizeOutput2 = new VectorUInt32(3);
                sizeOutput2.Add(256);
                sizeOutput2.Add(256);
                sizeOutput2.Add(256);
                constantImageSource2.SetOrigin(origin);
                constantImageSource2.SetSpacing(spacing);
                constantImageSource2.SetSize(sizeOutput2);
                constantImageSource2.SetConstant(0.0);
                Image source2 = constantImageSource2.Execute();

                Console.WriteLine("Performing reconstruction");
                FDKConeBeamReconstructionFilter feldkamp = new FDKConeBeamReconstructionFilter();
                feldkamp.SetGeometry(geometry);
                feldkamp.SetTruncationCorrection(0.0);
                feldkamp.SetHannCutFrequency(0.0);
                Image image = feldkamp.Execute(source2, reiImage);

                ImageFileWriter writer = new ImageFileWriter();
                writer.SetFileName(args[0]);
                writer.Execute(image);

            } catch (Exception ex) {
                Console.WriteLine(ex);
            }
        }
    }
}
