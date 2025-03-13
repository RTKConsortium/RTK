 # Create a simulated geometry
 rtksimulatedgeometry -n 180 -o geometry.xml
 # You may add "--arc 200" to make the scan short or "--proj_iso_x 200" to offset the detector

 # Create projections of the phantom file
 rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --dimension 256

 # Reconstruct
 rtkconjugategradient -p . -r projections.mha -o 3dcg.mha -g geometry.xml --spacing 2 --dimension 256 -n 10

 # Create a reference volume for comparison
 rtkdrawshepploganphantom --spacing 2 --dimension 256 -o ref.mha