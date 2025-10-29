# Amsterdam Shroud

The Amsterdam Shroud is an algorithm developed by Lambert Zijp (described [here](https://www.creatis.insa-lyon.fr/~srit/biblio/rit2012.pdf)) to extract a respiratory signal from the projection images of cone beam acquisitions of the free breathing thorax. This example is illustrated with a set of projection images of the [POPI patient](https://github.com/open-vv/popi-model/blob/master/popi-model.md). You can download the projections [here](https://data.kitware.com/api/v1/item/5be99af88d777f2179a2e144/download).

The Amsterdam Shroud tracks the diaphragm, which separates the lungs (low attenuation) and the abdomen (attenuation similar to water), by leveraging the strong attenuation contrast between these two regions.

It starts by generating a shroud image, in which each column stems from one projection, and is obtained by summing all columns of that projection.

```
rtkamsterdamshroud --path . \
                   --regexp '.*.his' \
                   --output shroud.mha \
                   --unsharp 650
```

![Signal](../../documentation/docs/ExternalData/shroud.png){w=800px alt="Shroud of POPI patient"}

Then it extracts a pseudo-periodic signal corresponding to the position of the diaphragm, and its phase.

```
rtkextractshroudsignal --input shroud.mha \
                       --output signal.txt \
                       --phase sphase.txt
```

Phase is commonly measured in radians, with values in $[0,2\pi[$, but in RTK it is normalized to $[0,1[$, where 0.3 corresponds to 30% in the respiratory cycle, i.e., frame 3 if you have a 10-frames 4D reconstruction or frame 6 if you have a 20-frames 4D reconstruction. The [resulting phase](https://data.kitware.com/api/v1/item/5be99af98d777f2179a2e160/download) is in green on top of the blue respiratory signal and the detected end-exhale peaks:

![Signal](../../documentation/docs/ExternalData/Signal.jpg){w=800px alt="Phase signal"}
