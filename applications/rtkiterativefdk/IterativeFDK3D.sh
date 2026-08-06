#!/usr/bin/env bash
# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml
# You may add "--arc 200" to make the scan short or "--proj_iso_x 200" to offset the detector

# Create projections of the phantom file
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 200

# Iterative reconstruction
rtkiterativefdk -p . -r projections.mha -o iterfdk.mha -g geometry.xml --spacing 2 --size 128 -n 5 --lambda 0.3 --positivity

# Create a reference volume for comparison
rtkdrawshepploganphantom --like iterfdk.mha -o ref.mha
