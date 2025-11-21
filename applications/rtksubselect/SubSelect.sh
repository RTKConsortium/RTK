# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml

# Create projections of the Shepp–Logan phantom
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 200

# Subselect every 2nd projection and write matching geometry
rtksubselect -p . -r projections.mha --geometry geometry.xml --out-geometry geometry_subset.xml --out-proj projections_subset.mha --first 30 --last 140 --step 2
