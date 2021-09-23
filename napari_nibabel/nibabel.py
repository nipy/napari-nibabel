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
from nibabel.orientations import (io_orientation, inv_ornt_aff,
                                  apply_orientation)

valid_volume_exts = {klass.valid_exts for klass in all_image_classes}
valid_volume_exts = set(functools.reduce(operator.add, valid_volume_exts))


def reorder_axes_to_ras(affine, data):
    """Permutes data dimensions and updates the affine accordingly.

    Reorders and/or flips data axes to get the spatial axes in RAS+ order.
    For oblique scans the axes closest to RAS (Right-Anterior-Superior) as
    determined by ``nibabel.affine.io_orientation``.

    Parameters
    ----------
    affine : (4, 4) ndarray
        Affine matrix
    data : ndarray
        Data array. Spatial dimensions must be first.

    Returns
    -------
    affine_ras : (4, 4) ndarray
        The affine matrix for the RAS-space data.
    data_ras : (4, 4) ndarray
        Data with axes reordered and/or flipped to RAS+ order.

    Notes
    -----
    Adapted from code converting to LAS+ in nibabel's parrec2nii.py
    """

    # Reorient data block to RAS+ if necessary
    ornt = io_orientation(affine)
    if np.all(ornt == [[0, 1],
                       [1, 1],
                       [2, 1]]):
        # already in desired orientation
        return affine, data

    # Reorient to RAS+
    t_aff = inv_ornt_aff(ornt, data.shape)
    affine_ras = np.dot(affine, t_aff)

    ornt = np.asarray(ornt)
    data_ras = apply_orientation(data, ornt)
    return affine_ras, data_ras


def adjust_translation(affine, affine_plumb, data_shape):
    """Adjust translation vector of affine_plumb.

    The goal is to have affine_plumb result in the same data center
    point in world coordinates as the original affine.

    Parameters
    ----------
    affine : ndarray
        The shape (4, 4) affine matrix read in by nibabel.
    affine_plumb: ndarray
        The affine after permutation to RAS+ space followed by discarding
        of any rotation/shear elements.
    data_shape : tuple of int
        The shape of the data array

    Returns
    -------
    affine_plumb : ndarray
        A copy of affine_plumb with the 3 translation elements updated.
    """
    data_shape = data_shape[-3:]
    if len(data_shape) < 3:
        # TODO: prepend or append?
        data_shape = data_shape + (1,) * (3 - data.ndim)

    # get center in world coordinates for the original RAS+ affine
    center_ijk = (np.array(data_shape) - 1) / 2
    center_world = np.dot(affine[:3, :3], center_ijk) + affine[:3, 3]

    # make a copy to avoid in-place modification of affine_plumb
    affine_plumb = affine_plumb.copy()

    # center in world coordinates with the current affine_plumb
    center_world_plumb =  np.dot(affine_plumb[:3, :3], center_ijk)

    # adjust the translation elements
    affine_plumb[:3, 3] = center_world - center_world_plumb
    return affine_plumb


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
    if not ext.lower() in valid_volume_exts:
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
        if not all([_obj.shape == objects[0].shape for _obj in objects]):
            raise ValueError(
                "all selected files must contain data of the same shape")

        if not all(np.allclose(affine, _obj.affine) for _obj in objects):
            raise ValueError(
                "all selected files must share a common affine")

        arrays = [_obj.get_fdata() for _obj in objects]

        # apply same transform to all volumes in the stack
        affine_orig = affine.copy()
        for i, arr in enumerate(arrays):
            affine, arr_ras = reorder_axes_to_ras(affine_orig, arr)
            arrays[i] = arr_ras

        # stack arrays into single array
        data = np.stack(arrays)
    else:
        img = nib.load(paths[0])
        header = img.header
        affine = img.affine
        data = img.get_fdata()  # keep this as dataobj or use get_fdata()?

        affine, data = reorder_axes_to_ras(affine, data)

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

    if np.all(affine[:3, :3] == (np.eye(3) * affine[:3, :3])):
        # no rotation or shear components
        affine_plumb = affine
    else:
        # Set any remaining non-diagonal elements of the affine to 0
        # (napari currently cannot display with rotate/shear)
        affine_plumb = np.diag(np.diag(affine))

        # Set translation elements of affine_plumb to get the center of the
        # data cube in the same position in world coordinates
        affine_plumb = adjust_translation(affine, affine_plumb, data.shape)

    # Note: The translate, scale, rotate, shear kwargs correspond to the
    # 'data2physical' component of a composite affine transform.
    # https://github.com/napari/napari/blob/v0.4.11/napari/layers/base/base.py#L254-L268   #noqa
    # However, the affine kwarg corresponds instead to the 'physical2world'
    # affine. Here, we will extract the scale and translate components from
    # affine_plumb so that we are specifying 'data2physical' to napari.

    add_kwargs = dict(
        metadata=dict(affine=affine, header=header),
        rgb=False,
        scale=np.diag(affine_plumb[:3, :3]),
        translate=affine_plumb[:3, 3],
        affine=None,
        channel_axis=None,
    )

    return [(data, add_kwargs, "image")]
