import atexit
import os
import shutil
import tempfile

import nibabel as nib
import numpy as np
import pytest

from napari_nibabel import napari_get_reader
from nibabel.testing import data_path


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data in NIFTI-1 format
    my_test_file = str(tmp_path / "myfile.nii")
    original_data = np.random.rand(20, 20, 1)

    # Set affine to an LPS affine here so internal reorientation will not be
    # needed.
    nii = nib.Nifti1Image(original_data, affine=np.diag((-1, -1, 1, 1)))
    nii.to_filename(my_test_file)
    np.save(my_test_file, original_data)

    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started
    np.testing.assert_allclose(original_data, layer_data_tuple[0])


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None


def _test_basic_read(filename):
    # try to read it back in
    reader = napari_get_reader(filename)

    # make sure we're delivering the right format
    layer_data_list = reader(filename)
    data0 = layer_data_list[0][0]

    assert callable(reader)
    return data0


def test_read_gzipped_nifti1_4d():
    filename = os.path.join(data_path, 'example4d.nii.gz')

    data = _test_basic_read(filename)
    assert data.ndim == 4
    assert data.shape[0] == 2  # this 4D example data has 2 timepoints


def test_read_gzipped_nifti2_4d():
    filename = os.path.join(data_path, 'example_nifti2.nii.gz')

    data = _test_basic_read(filename)
    assert data.ndim == 4
    assert data.shape[0] == 2  # this 4D example data has 2 timepoints


def test_read_nifti1_3d():
    filename = os.path.join(data_path, 'anatomical.nii')
    data = _test_basic_read(filename)
    assert data.ndim == 3  # this example data is a 3D volume


@pytest.mark.parametrize('ext', ['.PAR', '.REC'])
def test_read_Philips_parrec(ext):
    # can read a PAR/REC pair from either the .PAR or .REC file
    filename = os.path.join(data_path, 'phantom_EPI_asc_CLEAR_2_1' + ext)
    data = _test_basic_read(filename)
    assert data.ndim == 4


@pytest.mark.parametrize(
    'analyze_img_class',
    [nib.AnalyzeImage,
     nib.Spm2AnalyzeImage,
     nib.Spm99AnalyzeImage])
def test_read_analyze(tmp_path, analyze_img_class):
    # read sample NIFTI data and write it back out to analyze format
    nii_filename = os.path.join(data_path, 'anatomical.nii')
    nii = nib.load(nii_filename)
    img = analyze_img_class(nii.get_fdata(), affine=np.eye(4))
    img_filename = str(tmp_path / "anatomical.img")
    img.to_filename(img_filename)

    # read the generated ANALYZE file
    data = _test_basic_read(img_filename)
    assert data.ndim == 3

    # read via the header filename instead
    data = _test_basic_read(img_filename.replace('.img', '.hdr'))
    assert data.ndim == 3


def test_read_parrec_par_only():
    # read a .PAR file that does not have a corresponding .REC file
    filename = os.path.join(data_path, 'T1_3echo_mag_real_imag_phase.PAR')
    with pytest.raises(FileNotFoundError):
        _test_basic_read(filename)


@pytest.mark.parametrize('ext', ['.HEAD', '.BRIK'])
def test_read_AFNI(ext):
    # can read a .HEAD/.BRIK pair from either the .HEAD or .BRIK file
    filename = os.path.join(data_path, 'scaled+tlrc' + ext)
    data = _test_basic_read(filename)
    assert data.ndim == 4
    assert data.shape[0] == 1


@pytest.mark.parametrize('ext', ['.HEAD', '.BRIK.gz'])
def test_read_gzipped_AFNI_4d(ext):
    # can read a .HEAD/.BRIK.gz pair from either file
    filename = os.path.join(data_path, 'example4d+orig' + ext)
    data = _test_basic_read(filename)
    assert data.ndim == 4


# Note: tmp_path is a pytest fixture
def test_read_AFNI_head_only(tmp_path):
    # check error type when reading a .HEAD file without a corresponding .BRIK
    orig_filename = os.path.join(data_path, 'example4d+orig.HEAD')

    # copy .HEAD file to a location that doesn't contain the .BRIK data
    my_test_file = str(tmp_path / "example_4d_temp.HEAD")
    shutil.copyfile(orig_filename, my_test_file)
    with pytest.raises(FileNotFoundError):
        _test_basic_read(my_test_file)


def test_read_mnc_small():
    filename = os.path.join(data_path, 'small.mnc')
    data = _test_basic_read(filename)
    assert data.ndim == 3


def test_read_minc1_4d():
    filename = os.path.join(data_path, 'minc2_4d.mnc')
    data = _test_basic_read(filename)
    assert data.ndim == 4


def test_read_minc2_4d():
    filename = os.path.join(data_path, 'minc2_4d.mnc')
    data = _test_basic_read(filename)
    assert data.ndim == 4


def test_read_mgz():
    filename = os.path.join(data_path, 'test.mgz')
    data = _test_basic_read(filename)
    assert data.ndim == 4


def test_analyze_hdr_only():
    # read a .hdr file that does not have a corresponding .img file
    filename = os.path.join(data_path, 'analyze.hdr')
    with pytest.raises(FileNotFoundError):
        _test_basic_read(filename)


def test_read_filelist():
    filename = os.path.join(data_path, 'example4d.nii.gz')
    n_files = 3
    data = _test_basic_read([filename,]  * n_files)
    assert data.ndim == 5
    assert data.shape[0] == n_files


def test_read_filelist_mismatched_shape():
    # cannot stack multiple files when the shapes are different
    filename = os.path.join(data_path, 'example_nifti2.nii.gz')
    filename2 = os.path.join(data_path, 'example4d.nii.gz')
    with pytest.raises(ValueError):
        _test_basic_read([filename, filename2])


def test_read_filelist_mismatched_affine():
    # cannot stack multiple files when the shapes are different
    tmp_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, tmp_dir)

    filename = os.path.join(data_path, 'anatomical.nii')
    nii1 = nib.load(filename)
    data = nii1.get_fdata()
    affine2 = nii1.affine.copy()
    affine2[0, 0] *= 2
    affine2[1, 1] *= -1
    nii2 = nib.Nifti1Image(data, affine=affine2, header=nii1.header)
    filename2 = os.path.join(tmp_dir, 'anatomical_affine2.nii')
    nii2.to_filename(filename2)

    with pytest.raises(ValueError):
        _test_basic_read([filename, filename2])
