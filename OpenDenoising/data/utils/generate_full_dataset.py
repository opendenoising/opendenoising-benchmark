# Copyright or © or Copr. IETR/INSA Rennes (2019)
# 
# Contributors :
#     Eduardo Fernandes-Montesuma eduardo.fernandes-montesuma@insa-rennes.fr (2019)
#     Florian Lemarchand florian.lemarchand@insa-rennes.fr (2019)
# 
# 
# OpenDenoising is a computer program whose purpose is to benchmark image
# restoration algorithms.
# 
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software. You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# 
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author, the holder of the
# economic rights, and the successive licensors have only  limited
# liability.
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-C license and that you accept its terms.


import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage import img_as_ubyte


def generate_full_dataset(reference_images_folder, output_path, noise_config, preprocessing, n_channels=1):
    """Reads each images in reference_images_folder, apply preprocessing functions, adds noise then saves it
    to the inputs folder.

    Notes
    -----
    We assume that your images are contained directly on 'reference_images_folder'. The generated images (after
    preprocessing) will be saved to output_path/ref, and the images after noise addition will be saved
    to output_path/in.

    Parameters
    ----------
    reference_images_folder : str
        String containing the path to the reference images.
    output_path : str
        String containing the path to save the generated images.
    preprocessing : list
        List of preprocessing functions, which will be applied to each image.
    noise_config : dict
        Dictionary whose keys are functions implementing the noise process, and the value is a list containing the noise
        function parameters. If you do not want to specify any parameters, your list should be empty.
    n_channels : int
        1 (grayscale), or 3 (RGB).

    Examples
    --------
    The following example generates a Dataset consisting of 40 x 40 patches corrupted with 25-gaussian noise.

    >>> from functools import partial
    >>> from OpenDenoising.data.utils import gaussian_noise
    >>> from OpenDenoising.data.utils import gen_patches
    >>> from OpenDenoising.data.utils import generate_full_dataset
    >>> PATH_TO_IMGS = "./tmp/BSDS500/Train/ref/"
    >>> PATH_TO_SAVE = "./tmp/Cropped_40_BSDS_Gauss_25/"
    >>> generate_full_dataset(PATH_TO_IMGS, PATH_TO_SAVE, noise_config={gaussian_noise: [25]},
    ...                       preprocessing=[partial(gen_patches, patch_size=40)], n_channels=1)
    """
    filenames = os.listdir(reference_images_folder)
    noise_functions = noise_config.keys()
    noise_args = [noise_config[noise_type] for noise_type in noise_config]

    try:
        os.makedirs(os.path.join(output_path, "in"))
    except FileExistsError:
        if len(os.listdir(os.path.join(output_path, "in"))):
            raise FileExistsError("Directory {} already exists and is not"
                                  " empty.".format(os.path.join(output_path, "in")))
    try:
        os.makedirs(os.path.join(output_path, "ref"))
    except FileExistsError:
        if len(os.listdir(os.path.join(output_path, "ref"))):
            raise FileExistsError("Directory {} already exists and is not"
                                  " empty.".format(os.path.join(output_path, "ref")))

    pgbar = tqdm(filenames, ascii=True)
    for filename in pgbar:
        filepath = os.path.join(reference_images_folder, filename)
        pgbar.set_description("Processing image from {}".format(filepath))
        reference_arr = imread(filepath)[0] if n_channels == 3 else imread(filepath)
        if reference_arr.ndim == 2: reference_arr = np.expand_dims(reference_arr, axis=-1)
        if reference_arr.dtype == 'uint8': reference_arr = reference_arr.astype('float64') / 255
        input_arr = reference_arr.copy()

        for noise_function, noise_arg in zip(noise_functions, noise_args):
            # Adds noise to the reference.
            input_arr = noise_function(input_arr, *noise_arg)

        for func in preprocessing:
            # Applies preprocessing functions in order
            input_arr, reference_arr = func(input_arr, reference_arr)

        input_arr = img_as_ubyte(np.clip(input_arr, 0, 1))
        reference_arr = img_as_ubyte(np.clip(reference_arr, 0, 1))

        if input_arr.ndim > 3:
            for i, (input_patch, reference_patch) in enumerate(zip(input_arr, reference_arr)):
                imsave(os.path.join(output_path, "in", str(i) + "_" + filename), np.squeeze(input_patch),
                       check_contrast=False)
                imsave(os.path.join(output_path, "ref", str(i) + "_" + filename), np.squeeze(reference_patch),
                       check_contrast=False)
        else:
            imsave(os.path.join(output_path, "in", filename), np.squeeze(input_arr),
                   check_contrast=False)
            imsave(os.path.join(output_path, "ref", filename), np.squeeze(reference_arr),
                   check_contrast=False)

