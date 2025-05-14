 # Create a simulated geometry
 rtksimulatedgeometry -n 180 -o geometry.xml
 # You may add "--arc 200" to make the scan short or "--proj_iso_x 200" to offset the detector

 # Create projections of the phantom file
 rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 256

 # Reconstruct
 rtkadmmtotalvariation -p . -r projections.mha -o admmtv.mha -g geometry.xml --spacing 2 --size 256 --alpha 1 --beta 1000 -n 3

 # Create a reference volume for comparison
 rtkdrawshepploganphantom --spacing 2 --size 256 -o ref.mha