# Contribute documentation improvements

Thank you for improving our documentation!

The sources of the RTK documentation are spread across the RTK source tree in multiple markdown (.md) files.
Edit existing documentation files, add new examples or documentation files, and open a merge request on the RTK repository.

## Add a new documentation page
To add a new page to the documentation, create a new .md file in the RTK source tree.
Write plain markdown text or use the [MyST](https://myst-parser.readthedocs.io/en/latest/index.html) syntax to include technical and scientific content.
Additional MyST syntax extensions can be enabled by adding them to the `myst_enable_extensions` variable in the sphinx `conf.py` configuration file.

The new documentation page will appear in the documentation after being referenced in a `toctree` element of an `index.md` file.
Index files are markdown files using [TOC tree](https://sphinx-doc-zh.readthedocs.io/en/latest/markup/toctree.html) elements to define the table of content and sections of the documentation.
Add a reference to your new page in an existing section by adding the path to your file, relative to the `index.md` file, in an existing TOC tree element.
You may also add new `toctree` elements and new `index.md` file as needed.

### Inline images
Images can be referenced with their URL in markdown, but this can delay their presentation in the page.
Another option is to store the images in the [RTK collection](https://data.kitware.com/?#collection/5a7706878d777f0649e04776/folder/5eaeaa2f9014a6d84e47adb3) on data.kitware.com and to push their content file (SHA512) to the RTK source tree.
Such images will be automatically downloaded when building the documentation and served as static files when navigating.
To add a new image:
1. Go to the [RTKDocumentationData](https://data.kitware.com/?#collection/5a7706878d777f0649e04776/folder/5eaeaa2f9014a6d84e47adb3) folder on data.kitware.com
2. Choose a folder or create a new one and click the "Upload here" button to drag & drop your images.
3. Once uploaded, click on the file to open its information page.
4. Under the "Files & Links" section click the information icon ("Show info") next to the file name
5. Download the key file from the SHA-512 section by pressing the key icon
6. Place this file in the RTK source tree next to the .md file that uses it

The image can be referenced in markdown, e.g. `![img_2D](SheppLogan-2D.png){w=200px alt="Shepp-Logan 2D image"}`

### Inline code
Code blocks from external files can also be included in the documentation pages using the [literalinclude](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-literalinclude) directive.
~~~
```{literalinclude} code.py
```
~~~

### Add an example
Create a new directory in the examples directory to place the example scripts.
Then add the content link files of the screenshots to presented in the documentation by following the instructions from the "Inline images" section above.
Include the images and scripts in a `README.md` file describing the example.
Add a reference to the README file in the `examples/index.md` to add the new example page to the documentation

### Graphs

ITK's pipeline mechanism, described in their [software guide](https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch3.html#x39-420003.5), is described in the documention with [Graphviz](https://graphviz.org), both in the Sphinx doc (e.g. in [Projectors.md](Projectors.md)) and in the Doxygen doc (e.g. in the [ProjectionsReader](https://www.openrtk.org/Doxygen/classrtk_1_1ProjectionsReader.html)).

## Preview changes

To preview documentation changes, build and serve the documentation locally:

1. Install sphinx and the required packages:
```bash
cd RTK/documentation/docs
pip install -r requirements.txt
```
2. Configure RTK with the CMake option `RTK_BUILD_SPHINX` turned on
3. Build the `sphinx_doc` target and open the `index.html` file generated in `RTK-build/sphinx/sphinx_build`

The documentation is generated in the RTK build tree by first copying the documentation sources. To make sure new files are included in the local documentation builds, check that they belong to the list of files and directories copied by the `copy_sources` target in `documentation/docs/CMakeLists.txt`.
