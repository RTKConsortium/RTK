# Varian Reconstruction

## Varian OBI Reconstruction

The first step before proceeding with reconstruction is to convert Varian's geometry into RTK's format using a command-line tool. Follow these simple steps:

### 1. Download Varian Dataset

Download the dataset from [Varian-data](https://data.kitware.com/api/v1/item/5be94de88d777f2179a24de0/download).

### 2. Convert Geometry

Run the application to convert Varian's geometry into RTK's format:

```bash
rtkvarianobigeometry \
  --xml_file ProjectionInfo.xml \
  --path Scan0/ \
  --regexp Proj_.*.hnd \
  -o geometry.xml
```

### 3. Reconstruct Using RTK Applications

Reconstruct a slice (e.g., slice 30) of the volume using the `rtkfdk` algorithm:

```bash
rtkfdk \
  --geometry geometry.xml \
  --regexp .*\.hnd \
  --path Scan0 \
  --output slice30.mha \
  --verbose \
  --spacing 0.25,0.25,0.25 \
  --dimension 1024,1,1024 \
  --origin -127.875,30,-127.875
```

### 4. Apply the FOV Filter

Apply the field-of-view (FOV) filter to discard everything outside the FOV:

```bash
rtkfieldofview \
  --geometry geometry.xml \
  --regexp .*\.hnd \
  --path Scan0 \
  --reconstruction slice30.mha \
  --output slice30.mha \
  --verbose
```

### 5. Visualize the Result

You can visualize the result using a viewer (e.g., VV). The resulting image should look like this:

![Varian](Varian.png){w=400px alt="Varian snapshot"}

---

## Varian ProBeam Reconstruction

Follow these steps for the Varian ProBeam format:

### 1. Download Dataset

Download the dataset from [Varian-ProBeam-data](https://data.kitware.com/api/v1/item/5be94bef8d777f2179a24ae1/download).

### 2. Convert Geometry

Run the application to convert Varian ProBeam's geometry into RTK's format:

```bash
rtkvarianprobeamgeometry \
  --xml_file Scan.xml \
  --path Acquisitions/733061622 \
  --regexp Proj_.*.xim \
  -o geometry.xml
```

### 3. Reconstruct Using RTK Applications

Reconstruct a slice (e.g., slice 58) of the volume using the `rtkfdk` algorithm:

```bash
rtkfdk \
  --geometry geometry.xml \
  --regexp .*\.xim \
  --path Acquisitions/733061622 \
  --output slice58.mha \
  --verbose \
  --spacing 0.25,0.25,0.25 \
  --dimension 1024,1,1024 \
  --origin -127.875,-58,-127.875
```

### 4. Apply the FOV Filter

Apply the field-of-view (FOV) filter to discard everything outside the FOV:

```bash
rtkfieldofview \
  --geometry geometry.xml \
  --regexp .*\.xim \
  --path Acquisitions/733061622 \
  --reconstruction slice58.mha \
  --output slice58.mha \
  --verbose
```

### 5. Visualize the Result

You can visualize the result using a viewer (e.g., VV). The resulting image should look like this:

![VarianProBeam](VarianProBeam.png){w=400px alt="VarianProBeam snapshot"}