"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
import functools
import operator

import numpy as np
import nibabel as nib

from napari_plugin_engine import napari_hook_implementation

from nibabel.imageclasses import all_image_classes
from nibabel.filename_parser import splitext_addext


all_valid_exts = {klass.valid_exts for klass in all_image_classes}
all_valid_exts = set(functools.reduce(operator.add, all_valid_exts))

@napari_hook_implementation
def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    froot, ext, addext = splitext_addext(path)

    # if we know we cannot read the file, we immediately return None.
    if not ext.lower() in all_valid_exts:
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    n_spatial = 3
    # note: we don't squeeze the data below, so 2D data will be 3D with 1 slice
    if len(paths) > 1:
        # load all files into a single array
        objects = [nib.load(_path) for _path in paths]
        header = objects[0].header
        affine = objects[0].affine
        if not all([_obj.shape == _obj[0].shape for _obj in objects]):
            raise ValueError(
                "all selected files must contain data of the same shape")

        arrays = [_obj.get_fdata() for _obj in objects]

        # stack arrays into single array
        data = np.stack(arrays)
    else:
        img = nib.load(paths[0])
        header = img.header
        affine = img.affine
        data = img.get_fdata()  # keep this as dataobj or use get_fdata()?

        spatial_axis_order = tuple(range(n_spatial))
        if data.ndim > 3:
            # nibabel formats have spatial axes in the first 3 positions, but
            # we need to move these to the last 3 for napari.
            axes = tuple(range(data.ndim))
            n_nonspatial = data.ndim - 3
            new_axis_order = axes[-1:-n_nonspatial - 1:-1] + spatial_axis_order
            data = data.transpose(new_axis_order)
        else:
            if spatial_axis_order != (0, 1, 2):
                data = data.transpose(spatial_axis_order[:data.ndim])

    try:
        # only get zooms for the spatial axes
        zooms = np.asarray(header.get_zooms())[:n_spatial]
        if np.any(zooms == 0):
            raise ValueError("invalid zoom = 0 found in header")
        # normalize so values are all >= 1.0 (not strictly necessary)
        # zooms = zooms / zooms.min()
        zooms = tuple(zooms)
        if data.ndim > 3:
            zooms = (1.0, ) * (data.ndim - n_spatial) + zooms
    except (AttributeError, ValueError):
        zooms = (1.0, ) * data.ndim

    apply_translation = False
    if apply_translation:
        translate = tuple(affine[:n_spatial, 3])
        if data.ndim > 3:
            # set translate = 0.0 on non-spatial dimensions
            translate = (0.0,) * (data.ndim - n_spatial) + translate
    else:
        translate = (0.0,) * data.ndim

    # optional kwargs for the corresponding viewer.add_* method
    # https://napari.org/docs/api/napari.components.html#module-napari.components.add_layers_mixin
    # see also: https://napari.org/tutorials/fundamentals/image
    add_kwargs = dict(
        metadata=dict(affine=affine, header=header),
        rgb=False,
        scale=zooms,
        translate=translate,
        # contrast_limits=,
    )

    # TODO: potential kwargs to set for viewer.add_image
    #     contrast_limits kwarg based on info in image header?
    #          e.g. for NIFTI: nii.header._structarr['cal_min']
    #                          nii.header._structarr['cal_max']

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]
