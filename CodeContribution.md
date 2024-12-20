# Code contribution

## Coding style

RTK is based on ITK and aims at following its coding conventions. Any developer should follow these conventions when submitting new code or contributions to the existing one. We strongly recommend you to read thoroughly [ITK's style guide](http://www.itk.org/Wiki/ITK/Coding_Style_Guide).

## Testing

This section describes how to add/edit datasets for testing purposes for RTK. Datasets are not stored in the GIT repository for efficiency and also to avoid having large history due to binary files. Instead the files are stored on a [Girder](http://data.kitware.com) instance. Here's the recipe to add new datasets:

1.  Register/Login to Girder hosted at Kitware: [http://data.kitware.com](http://data.kitware.com)
2.  Locate the RTK collection: [https://data.kitware.com/#collection/5a7706878d777f0649e04776](https://data.kitware.com/#collection/5a7706878d777f0649e04776)
3.  Upload the new datasets in the appropriate folder. If you don't have the necessary privileges please email the mailing list
4.  In the GIT repository, add in testing/Data a file with the exact filename of the original file **but with the .md5 extension**. Inside that file put the md5sum of the file on Girder.
5.  When adding a test use the new macro 'RTK_ADD_TEST' instead of 'ADD_TEST' and specify the datasets you want CTest to download by appending the data to 'DATA{}'. For example:

```
RTK_ADD_TEST(NAME rtkimagxtest
    COMMAND ${EXECUTABLE\_OUTPUT\_PATH}/rtkimagxtest
    DATA{Data/Input/ImagX/raw.xml,raw.raw}
    DATA{Data/Baseline/ImagX/attenuation.mha})
```
## Dashboard

*   The RTK dashboard is available at [RTK Dashboard](http://my.cdash.org/index.php?project=RTK)