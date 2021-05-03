"""
This module is part of the napari-nibabel plugin.
It contains reader functions for the file formats supported by nibabel.
The reader passes metainformation stored in the file headers as well as spatial to the viewer.

The script is based on the napari cookiecutter template:
https://github.com/napari/cookiecutter-napari-plugin
"""
import functools
import operator

import numpy as np
import nibabel as nib

from napari_plugin_engine import napari_hook_implementation

from nibabel.imageclasses import all_image_classes
from nibabel.filename_parser import splitext_addext
from nibabel.orientations import io_orientation, ornt_transform

all_valid_exts = {klass.valid_exts for klass in all_image_classes}
all_valid_exts = set(functools.reduce(operator.add, all_valid_exts))

# Napari's standard dimension order is z, y, x with the axe origin in the upper left corner.
# To display te images in the viewer in a radiological way they must align to the orientation:
# z: I > S, y: A > P, x: R > L
viewer_ornt = np.array([[2., 1.],
                        [1., -1.],
                        [0., -1.]],
                       dtype=int)


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

    # note: we don't squeeze the data below, so 2D data will be 3D with 1 slice
    if len(paths) > 1:
        # combina all images into one
        objects = [nib.load(_path) for _path in paths]
        imgobj = nib.concat_images(objects)
    else:
        imgobj = nib.load(paths[0])
    # transform the images to the viewer orientation
    img_ornt = io_orientation(imgobj.affine)
    t_ornt = ornt_transform(img_ornt, viewer_ornt)
    t_imgobj = imgobj.as_reoriented(t_ornt)

    # keep this as dataobj or use get_fdata()?
    data = t_imgobj.get_fdata()
    # hande 2D images as 3D images with one slice
    if data.ndim < 3:
        data = np.expand_dims(data, axis=2)
    header = t_imgobj.header
    affine = t_imgobj.affine

    # handle 4D images, bring the temporal axis to the front and keep the spatial ordering
    if data.ndim == 4:
        data = np.transpose(data, (3, 0, 1, 2))

    # TODO: At present napari doesn't fully support non-orthogonal slicing. Thus, the 2d view is not usable,
    #  with affine transformation including out of slice rotation or similar.
    #  To preserve the 2D viewer functionality we have to wait till napari fully supports the operations,
    #  befor adding affine transformation.

    # generate scale values
    try:
        # get the zooms from image metadata
        zooms = header.get_zooms()[:3]
        if any([i == 0 for i in zooms]):
            raise ValueError("invalid zoom = 0 found in header")
        # normalize so values are all >= 1.0 (not strictly necessary)
        # zooms = zooms / zooms.min()
        if data.ndim > 3:
            zooms = (1.0, ) * (data.ndim - 3) + zooms
    except (AttributeError, ValueError):
        zooms = (1.0, ) * data.ndim

    # TODO: why not apply translate?

    apply_translation = True
    if apply_translation:
        # get translate from affine
        translate = affine[:3, 3]
        if data.ndim > 3:
            # set translate = 0.0 on non-spatial dimensions
            translate = (0.0, ) * (data.ndim - 3) + translate
    else:
        translate = (0.0, ) * data.ndim

    add_kwargs = dict(
        metadata=dict(imgobj=imgobj),
        rgb=False,
        scale=zooms,
        translate=translate,
        # contrast_limits=,
    )

    # TODO: potential kwargs to set for viewer.add_image
    #     contrast_limits kwarg based on info in image header?
    #          e.g. for NIFTI: nii.header._structarr['cal_min']
    #                          nii.header._structarr['cal_max']

    layer_type = 'image'
    # TODO: maybe add a detection for label images, to load as labels layer?
    return [(data, add_kwargs, layer_type)] 
