# Elekta Reconstruction

Elekta provides easy access to raw data. The data and projection images are stored in a single directory which is user-configurable. The default location is `D:\db`. In this folder, there is a database in DBase format. Each table is contained in a `.DBF` file. RTK needs the `IMAGE.DBF` and `FRAME.DBF` tables.

Patient data are stored in individual folders. By default, the name of each patient folder is `patient_ID` where `ID` is the patient ID. In these folders, one can access the planning CT in the `CT_SET` subfolder and the cone-beam projections in `IMAGES/img_DICOM_UID` subfolders, where `DICOM_UID` is the DICOM UID of the acquisition. The projection images are `.his` files. The reconstructed images are the `IMAGES/img_DICOM_UID/Reconstruction/*SCAN` files.

## Reconstruction Steps

The first step before proceeding with the reconstruction is to convert Elekta's database information into an RTK geometry file using a command-line tool. Follow these steps:

### 1. Download Elekta Dataset

Download the dataset from [Elekta-data](https://data.kitware.com/api/v1/item/5be973478d777f2179a26e1c/download).

### 2. Convert Geometry

Run the application to convert Elekta's geometry into RTK's format (DICOM_UID is contained in the subfolder name of the `.his` files):

```bash
rtkelektasynergygeometry \
  --image_db IMAGE.DBF \
  --frame_db FRAME.DBF \
  --dicom_uid 1.3.46.423632.135428.1351013645.166 \
  -o elektaGeometry
```

Since XVI v5, the geometry is contained in a separate `_Frames.xml` file, which can be used as follows:

```bash
rtkelektasynergygeometry \
  --xml _Frames.xml \
  -o elektaGeometry
```

An example of such a file is available in the test data [here](https://data.kitware.com/api/v1/item/5b179c898d777f15ebe201fd/download).

### 3. Reconstruct Using RTK Applications

Use the `rtkfdk` algorithm to reconstruct a single axial slice (e.g., slice 29.5) of the volume:

```bash
rtkfdk \
  --lowmem \
  --geometry elektaGeometry \
  --path img_1.3.46.423632.135428.1351013645.166/ \
  --regexp '.*.his' \
  --output slice29.5.mha \
  --verbose \
  --spacing 0.25,0.25,0.25 \
  --size 1024,1,1024 \
  --origin -127.875,29.5,-127.875
```

### 4. Apply the FOV Filter

Apply the field-of-view (FOV) filter to mask out everything outside the FOV:

```bash
rtkfieldofview \
  --geometry elektaGeometry \
  --path img_1.3.46.423632.135428.1351013645.166/ \
  --regexp '.*.his' \
  --reconstruction slice29.5.mha \
  --output slice29.5.mha \
  --verbose
```

### 5. Visualize the Result

You can visualize the result using a viewer (e.g., VV). The resulting image should look like the following:

![Elekta.jpg](../../documentation/docs/ExternalData/Elekta.png){w=400px alt="Elekta snapshot"}
