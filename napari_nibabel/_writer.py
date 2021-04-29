"""
This module is part of the napari-nibabel plugin.
It contains writer functions for image layers.
The layers must contain a nibabel image object in its metadata.

The script is based on the napari cookiecutter template:
https://github.com/napari/cookiecutter-napari-plugin
"""

import functools
import operator
import warnings

import nibabel as nib
import numpy as np

from napari_plugin_engine import napari_hook_implementation
from napari.types import LayerData

from nibabel.imageclasses import all_image_classes
from nibabel.filename_parser import splitext_addext

from typing import List, Dict

all_valid_exts = {klass.valid_exts for klass in all_image_classes}
all_valid_exts = set(functools.reduce(operator.add, all_valid_exts))
all_valid_layers = {'image', }

# TODO: support more layer types
# TODO: add function to build nibabel.SpatialImage from blank array and file extension


@napari_hook_implementation
def napari_get_writer(path: str, layer_types: List[str]) -> callable or None:
    """Retrun writer function if the file format (all nibabel formats) and layer types (image, labels) are suppported.

    :param path: Path to file.
    :param layer_types: List of layer types presented to the writer,
    only layer types included in the supported_layers list will return a writer function other will return None.
    :return: If the layer type and file extension is supported return the nifti_writer function.
    """
    # Return None if not supported layer types are selected
    if not all(lt in all_valid_layers for lt in layer_types):
        return None

    froot, ext, addext = splitext_addext(path)

    # Return None if the requested file format is not supported
    if not ext.lower() in all_valid_exts:
        return None

    return multilayer_writer


def multilayer_writer(path: str, layer_data: List[LayerData]) -> str or None:
    """Write supported layers to files.

    :param path: Path to file or directory.
    :param layer_data: List of napari.types.LayerData of shape Tuple(data, meta, layer_type).
    :return: If layer was successfully written return the save path otherwise None.
    """
    paths = []

    for (data, meta, lt) in layer_data:

        # Generate path per layer
        froot, ext, addext = splitext_addext(path)
        layer_path = '.'.join(['_'.join([froot, meta['name']]), ext, addext])

        # Return None if no nibabel image object is in metadata
        try:
            imgobj = meta['metadata']['imgobj']
        except KeyError:
            warnings.warn('No nibabel image object detected in metadata. Aborting.')
            return None

        # Write data to file
        nib.save(layer_path, imgobj)
        paths.append(layer_path)

    return paths


@napari_hook_implementation
def napari_write_image(path: str, data: np.ndarray, meta: Dict) -> str or None:
    """Write image layer to file.

    :param path: Path to file or directory.
    :param data: Layer data.
    :param meta: All metadata of the layer (name, scale, metadata,...)
    :return: If layer was successfully written return the save path otherwise None.
    """
    froot, ext, addext = splitext_addext(path)

    # Return None if the requested file format is not supported
    if not ext.lower() in all_valid_exts:
        return None

    # Return None if no nibabel image object is in metadata
    try:
        imgobj = meta['metadata']['imgobj']
    except KeyError:
        warnings.warn('No nibabel.SpatialImage detected in metadata. Aborting.')
        return None

    # Write data to file
    nib.save(path, imgobj)

    return path
