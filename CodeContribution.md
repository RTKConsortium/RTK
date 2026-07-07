# Code contribution

## Coding style

RTK is based on ITK and aims at following its coding conventions. Any developer should follow these conventions when submitting new code or contributions to the existing one. We strongly recommend you reading thoroughly [ITK's style guide](http://www.itk.org/Wiki/ITK/Coding_Style_Guide).

All (new) command-line options in applications must use hyphen-separated lowercase names (for example, `output-file`), following the [glibc convention](https://ftp.gnu.org/old-gnu/Manuals/glibc-2.2.3/html_node/libc_511.html). This rule applies to both C++ and Python applications. Some existing options not following this style are kept for backward compatibility.

## Testing

This section describes how to add/edit datasets for testing purposes for RTK. Datasets are not stored in the GIT repository for efficiency and also to avoid having large history due to binary files. Instead, the files are stored on a [Girder](http://data.kitware.com) instance. Here's the recipe to add new datasets:

1.  Register/Login to Girder hosted at Kitware: [http://data.kitware.com](http://data.kitware.com)
2.  Locate the RTK collection: [https://data.kitware.com/#collection/5a7706878d777f0649e04776](https://data.kitware.com/#collection/5a7706878d777f0649e04776)
3.  Upload the new datasets in the appropriate folder. If you do not have the necessary privileges please email the mailing list
4.  In the GIT repository, add a file in `test/Baseline` or `test/Input` a file with the exact filename of the original file **but with the .md5 extension**. Inside that file put the md5sum of the file on Girder.
5.  When adding a test use the new macro `rtk_add_test` instead of `itk_add_test` and specify the datasets you want CTest to download by appending the data to `DATA{}`. For example:
```cmake
rtk_add_test(NAME rtkimagxtest
    COMMAND ${EXECUTABLE\_OUTPUT\_PATH}/rtkimagxtest
    DATA{Data/Input/ImagX/raw.xml,raw.raw}
    DATA{Data/Baseline/ImagX/attenuation.mha})
```
Alternatively, tests can also be defined in Python. RTK uses [`pytest`](https://docs.pytest.org/en/stable/) for the Python tests and follows its conventions. To add a Python test, create a Python file in the `test` folder. The name of the file should end in `_test.py` in order to be automatically picked up by `pytest`. Then create a test function whose name starts with `test_`. This function will be automatically executed when running `pytest`. For example:
```python
def test_IsIntersectedByRay():
    q = rtk.QuadricShape.New()
    radius = 10
    q.SetA(1)
    q.SetB(1)
    q.SetC(1)
    q.SetJ(-(radius**2))
    res = q.IsIntersectedByRay([-20, 0, 0], [1, 0, 0])
    assert isinstance(res, (list, tuple)) and len(res) >= 3
```

## Dashboard

*   The RTK dashboard is available at [RTK Dashboard](http://my.cdash.org/index.php?project=RTK)
