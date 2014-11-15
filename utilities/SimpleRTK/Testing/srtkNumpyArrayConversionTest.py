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
from __future__ import print_function
import sys
import unittest


import SimpleRTK as srtk
import numpy as np

sizeX = 4
sizeY = 5
sizeZ = 3


class TestNumpySimpleRTKInterface(unittest.TestCase):
    """ This tests numpy array <-> SimpleRTK Image conversion. """


    def setUp(self):
        pass


    def _helper_check_srtk_to_numpy_type(self, srtkType, numpyType):
        image = srtk.Image( (9, 10), srtkType, 1 )
        a = srtk.GetArrayFromImage( image )
        self.assertEqual( numpyType, a.dtype )
        self.assertEqual( (10, 9), a.shape )

    def test_type_to_numpy(self):
        "try all srtk pixel type to convert to numpy"

        self._helper_check_srtk_to_numpy_type(srtk.srtkUInt8, np.uint8)
        self._helper_check_srtk_to_numpy_type(srtk.srtkUInt16, np.uint16)
        self._helper_check_srtk_to_numpy_type(srtk.srtkUInt32, np.uint32)
        if srtk.srtkUInt64 != srtk.srtkUnknown:
            self._helper_check_srtk_to_numpy_type(srtk.srtkUInt64, np.uint64)
        self._helper_check_srtk_to_numpy_type(srtk.srtkInt8, np.int8)
        self._helper_check_srtk_to_numpy_type(srtk.srtkInt16, np.int16)
        self._helper_check_srtk_to_numpy_type(srtk.srtkInt32, np.int32)
        if srtk.srtkInt64 != srtk.srtkUnknown:
            self._helper_check_srtk_to_numpy_type(srtk.srtkInt64, np.int64)
        self._helper_check_srtk_to_numpy_type(srtk.srtkFloat32, np.float32)
        self._helper_check_srtk_to_numpy_type(srtk.srtkFloat64, np.float64)
        #self._helper_check_srtk_to_numpy_type(srtk.srtkComplexFloat32, np.complex64)
        #self._helper_check_srtk_to_numpy_type(srtk.srtkComplexFloat64, np.complex128)
        self._helper_check_srtk_to_numpy_type(srtk.srtkVectorUInt8, np.uint8)
        self._helper_check_srtk_to_numpy_type(srtk.srtkVectorInt8, np.int8)
        self._helper_check_srtk_to_numpy_type(srtk.srtkVectorUInt16, np.uint16)
        self._helper_check_srtk_to_numpy_type(srtk.srtkVectorInt16, np.int16)
        self._helper_check_srtk_to_numpy_type(srtk.srtkVectorUInt32, np.uint32)
        self._helper_check_srtk_to_numpy_type(srtk.srtkVectorInt32, np.int32)
        if srtk.srtkVectorUInt64 != srtk.srtkUnknown:
            self._helper_check_srtk_to_numpy_type(srtk.srtkVectorUInt64, np.uint64)
        if srtk.srtkVectorInt64 != srtk.srtkUnknown:
            self._helper_check_srtk_to_numpy_type(srtk.srtkVectorInt64, np.int64)
        self._helper_check_srtk_to_numpy_type(srtk.srtkVectorFloat32, np.float32)
        self._helper_check_srtk_to_numpy_type(srtk.srtkVectorFloat64, np.float64)
        #self._helper_check_srtk_to_numpy_type(srtk.srtkLabelUInt8, np.uint8)
        #self._helper_check_srtk_to_numpy_type(srtk.srtkLabelUInt16, np.uint16)
        #self._helper_check_srtk_to_numpy_type(srtk.srtkLabelUInt32, np.uint32)
        #self._helper_check_srtk_to_numpy_type(srtk.srtkLabelUInt64, np.uint64)

    def test_to_numpy_and_back(self):
        """Test converting an image to numpy and back"""

        img = srtk.GaussianSource( srtk.srtkFloat32,  [100,100], sigma=[10]*3, mean = [50,50] )

        h = srtk.Hash( img )

        # convert the image to and fro a numpy array
        img = srtk.GetImageFromArray( srtk.GetArrayFromImage( img ) )

        self.assertEqual( h, srtk.Hash( img ))

    def test_vector_image_to_numpy(self):
        """Test converting back and forth between numpy and SimpleRTK
        images were the SimpleRTK image has multiple componets and
        stored as a VectorImage."""


        # Check 2D
        img = srtk.PhysicalPointSource(srtk.srtkVectorFloat32, [3,4])
        h = srtk.Hash( img )

        nda = srtk.GetArrayFromImage(img)

        self.assertEqual(nda.shape, (4,3,2))
        self.assertEqual(nda[0,0].tolist(), [0,0])
        self.assertEqual(nda[2,1].tolist(), [1,2])
        self.assertEqual(nda[0,:,0].tolist(), [0,1,2])

        img2 = srtk.GetImageFromArray( nda, isVector=True)
        self.assertEqual( h, srtk.Hash(img2) )

        # check 3D
        img = srtk.PhysicalPointSource(srtk.srtkVectorFloat32, [3,4,5])
        h = srtk.Hash( img )

        nda = srtk.GetArrayFromImage(img)

        self.assertEqual(nda.shape, (5,4,3,3))
        self.assertEqual(nda[0,0,0].tolist(), [0,0,0])
        self.assertEqual(nda[0,0,:,0].tolist(), [0,1,2])
        self.assertEqual(nda[0,:,1,1].tolist(), [0,1,2,3])


        img2 = srtk.GetImageFromArray(nda)
        self.assertEqual(img2.GetSize(), img.GetSize())
        self.assertEqual(img2.GetNumberOfComponentsPerPixel(), img.GetNumberOfComponentsPerPixel())
        self.assertEqual(h, srtk.Hash(img2))


    def test_legacy(self):
      """Test SimpleRTK Image to numpy array."""

      #     self.assertRaises(TypeError, srtk.GetArrayFromImage, 3)

      # 2D image
      image = srtk.Image(sizeX, sizeY, srtk.srtkInt32)
      for j in range(sizeY):
          for i in range(sizeX):
              image[i, j] = j*sizeX + i

      print(srtk.GetArrayFromImage(image))

      self.assertEqual( type (srtk.GetArrayFromImage(image)),  np.ndarray )

      # 3D image
      image = srtk.Image(sizeX, sizeY, sizeZ, srtk.srtkFloat32)
      for k in range(sizeZ):
          for j in range(sizeY):
              for i in range(sizeX):
                  image[i, j, k] = (sizeY*k +j)*sizeX + i

      print(srtk.GetArrayFromImage(image))

      self.assertEqual( type (srtk.GetArrayFromImage(image)),  np.ndarray )

    def test_legacy_array2srtk(self):
      """Test numpy array to SimpleRTK Image."""

      arr = np.arange(20, dtype=np.float64)
      arr.shape = (sizeY, sizeX)

      image = srtk.GetImageFromArray(arr)
      self.assertEqual(image.GetSize(), (sizeX, sizeY))
      self.assertEqual(image[0,0], 0.0)
      self.assertEqual(image[1,1], 5.0)
      self.assertEqual(image[2,2], 10.0)

      arr = np.arange(60, dtype=np.int16)
      arr.shape = (sizeZ, sizeY, sizeX)

      image = srtk.GetImageFromArray(arr)

      self.assertEqual(image.GetSize(), (sizeX, sizeY, sizeZ))
      self.assertEqual(image[0,0,0], 0)
      self.assertEqual(image[1,1,1], 25)
      self.assertEqual(image[2,2,2], 50)

if __name__ == '__main__':
    unittest.main()
