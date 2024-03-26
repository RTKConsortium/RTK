=====
 FDK
=====

Reconstruction of the Shepp–Logan phantom using Feldkamp, David and Kress cone-beam reconstruction.

3D
==
|sin_3D| |img_3D|

This script uses the file `SheppLogan.txt`_ as input.

.. literalinclude:: Code3D.sh

.. _SheppLogan.txt: https://data.kitware.com/api/v1/item/5b179c938d777f15ebe2020b/download

.. |sin_3D| image:: SheppLogan-3D-Sinogram.png
  :scale: 30%
  :alt: Shepp-Logan 3D sinogram

.. |img_3D| image:: SheppLogan-3D.png
  :scale: 30%
  :alt: Shepp-Logan 3D image

2D
==
|sin_2D| |img_2D|

The same reconstruction can be performed using the original 2D Shepp-Logan phantom.
RTK can perform 2D reconstructions through images wide of 1 pixel in the y direction.
The following script performs the same reconstruction as above in a 2D environment and uses the `2D Shepp-Logan`_ phantom as input.

.. literalinclude:: Code2D.sh

.. _2D Shepp-Logan: http://wiki.openrtk.org/images/7/73/SheppLogan-2d.txt

.. |sin_2D| image:: SheppLogan-2D-Sinogram.png
  :scale: 50%
  :alt: Shepp-Logan 2D sinogram

.. |img_2D| image:: SheppLogan-2D.png
  :scale: 50%
  :alt: Shepp-Logan 2D image