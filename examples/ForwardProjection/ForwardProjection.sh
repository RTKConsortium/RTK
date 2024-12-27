 # Create a simulated geometry
 rtksimulatedgeometry -n 180 -o geometry.xml

 # Forward project
 rtkforwardprojections -g geometry.xml -o projections.mha -i 00.mhd --spacing 2 --dimension 512

 # Reconstruct in the same resolution as the original
 rtkfdk -p . -r projections.mha -o fdk.mha -g geometry.xml --spacing=0.976562,0.976562,2 --origin=-250,-250,-164.5 --dimension=512,512,141
