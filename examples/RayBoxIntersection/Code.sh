 # Create a simulated geometry
 rtksimulatedgeometry -n 360 -o geometry.xml

 # Create a box volume
 rtkdrawgeometricphantom -o box.mha --spacing 1 --dimension 90 --phantomfile box.txt

 # Calculate radiological path of the box (ray intersection length)
 rtkrayboxintersection -g geometry.xml -i box.mha -o rayboxintersection.mha --spacing 1 --dimension 256

 # Reconstruct/Backproject the set of projections
 rtkfdk -g geometry.xml -p . -r rayboxintersection.mha -o fdk.mha --spacing 1 --dimension 256