# napari-nibabel

<!---
[![License](https://img.shields.io/pypi/l/napari-nibabel.svg?color=green)](https://github.com/napari/napari-nibabel/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-nibabel.svg?color=green)](https://pypi.org/project/napari-nibabel)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-nibabel.svg?color=green)](https://python.org)
--->
[![tests](https://github.com/nipy/napari-nibabel/workflows/tests/badge.svg)](https://github.com/nipy/napari-nibabel/actions)
[![codecov](https://codecov.io/gh/nipy/napari-nibabel/branch/master/graph/badge.svg)](https://codecov.io/gh/nipy/napari-nibabel)

A napari i/o plugin for neuroimaging formats (via nibabel/pydicom). Note that
this project is in an early prototype stage and is missing many features.

Currently, it should be able to open the following formats in Napari:

NIFTI1, NIFTI2, ANALYZE, Philips PAR/REC, MINC1, MINC2, AFNI BRIK/HEAD and
MGH/MGZ.

The following features are still pending:

- use affine information to orient and size the volumes properly
- use voxel dimensions so that anisotropic volumes will be displayed properly
- center the napari display at the coordinate origin upon opening
- add DICOM support
- add ECAT format support (PET)
- add file writing support

----------------------------------

## Installation

You can install `napari-nibabel` via [pip]:

    pip install napari-nibabel

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD 3-Clause] license, "napari-nibabel" is
free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed
description.


This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD 3-Clause]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/nipy/napari-nibabel/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/