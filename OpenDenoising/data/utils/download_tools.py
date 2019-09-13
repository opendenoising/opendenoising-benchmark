#Copyright or © or Copr. IETR/INSA Rennes (2019)
#
#Contributors :
#    Eduardo Fernandes-Montesuma eduardo.fernandes-montesuma@insa-rennes.fr (2019)
#    Florian Lemarchand florian.lemarchand@insa-rennes.fr (2019)
#
#
#OpenDenoising is a computer program whose purpose is to benchmark image
#restoration algorithms.
#
#This software is governed by the CeCILL-C license under French law and
#abiding by the rules of distribution of free software. You can  use,
#modify and/ or redistribute the software under the terms of the CeCILL-C
#license as circulated by CEA, CNRS and INRIA at the following URL
#"http://www.cecill.info".
#
#As a counterpart to the access to the source code and rights to copy,
#modify and redistribute granted by the license, users are provided only
#with a limited warranty  and the software's author, the holder of the
#economic rights, and the successive licensors have only  limited
#liability.
#
#In this respect, the user's attention is drawn to the risks associated
#with loading, using, modifying and/or developing or reproducing the
#software by the user in light of its specific status of free software,
#that may mean  that it is complicated to manipulate,  and  that  also
#therefore means  that it is reserved for developers  and  experienced
#professionals having in-depth computer knowledge. Users are therefore
#encouraged to load and test the software's suitability as regards their
#requirements in conditions enabling the security of their systems and/or
#data to be ensured and, more generally, to use and operate it in the
#same conditions as regards security.
#
#The fact that you are presently reading this means that you have had
#knowledge of the CeCILL-C license and that you accept its terms.


import os
import github
import numpy as np
import urllib.request

from tqdm import tqdm
from cv2 import imread, imwrite
from OpenDenoising.data import module_logger


def download_from_url(url, output_dir):
    """Download files from url. Saves them to output_dir.

    Parameters
    ----------
    url : str
        String containing the file's web address.
    output_dir : str
        String containing the path to the directory which files will be saved.
    """
    url_from = ''.join(url[0].split("/")[:-1])
    for img_url in tqdm(url, ascii=True, desc="Downloading images from {}".format(url_from)):
        filename = img_url.split("/")[-1]
        urllib.request.urlretrieve(url=img_url, filename=os.path.join(output_dir, filename))


def download_from_github(repo_addr, repo_dir, output_dir):
    """Downloads the content of a Github repository

    Parameters
    ----------
    repo_addr : str
        String containing the name of the repository containing the files to be downloaded.
    repo_dir : str
        String containing the name of the directory within the repository that contains the files to be downloaded.
    output_dir : str
        Directory where files will be saved.
    """
    # Github API
    g = github.Github()
    repo = g.get_repo(repo_addr)

    # Get repo contents
    contents = repo.get_contents(repo_dir)
    files_url = [content.download_url for content in contents]

    try:
        os.makedirs(output_dir)
    except FileExistsError as err:
        module_logger.warning("Directory {} already exists. Download was aborted. Check if the data"
                              " was already downloaded.".format(output_dir))
        raise err
    download_from_url(files_url, output_dir)


def download_PolyU(output_dir="./tmp/PolyU/"):
    """Downloads PolyU datast from the author's `Github page
    <https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset>`_.

    Parameters
    ----------
    output_dir : str
        String to the directory where images will be saved.
    """
    # Downloading train images
    process_train_data = True
    process_valid_data = True

    try:
        download_from_github(repo_addr="csjunxu/PolyU-Real-World-Noisy-Images-Dataset",
                             repo_dir="CroppedImages",
                             output_dir=os.path.join(output_dir, "Train"))
    except FileExistsError:
        process_train_data = False
    # Get all filenames
    filenames = os.listdir(os.path.join(output_dir, "Train"))

    # Creating partitions
    try:
        os.makedirs(os.path.join(output_dir, "Train", "in"))
    except FileExistsError:
        pass

    try:
        os.makedirs(os.path.join(output_dir, "Train", "ref"))
    except FileExistsError:
        pass

    # Dividing images between in/ref
    if process_train_data:
        for filename in filenames:
            _filename = filename
            partition = "ref" if 'real' in filename.lower() else 'in'
            _filename = _filename.replace("_real", "")
            _filename = _filename.replace("_mean", "")
            os.rename(os.path.join(output_dir, "Train", filename),
                      os.path.join(output_dir, "Train", partition, _filename))

    # Downloading train images
    try:
        download_from_github(repo_addr="csjunxu/PolyU-Real-World-Noisy-Images-Dataset",
                             repo_dir="OriginalImages",
                             output_dir=os.path.join(output_dir, "Valid"))
    except FileExistsError:
        process_valid_data = False
    # Get all filenames
    filenames = os.listdir(os.path.join(output_dir, "Valid"))

    # Creating partitions
    try:
        os.makedirs(os.path.join(output_dir, "Valid", "in"))
    except FileExistsError:
        pass

    try:
        os.makedirs(os.path.join(output_dir, "Valid", "ref"))
    except FileExistsError:
        pass

    # Dividing images between in/ref
    if process_valid_data:
        for filename in filenames:
            _filename = filename
            partition = "ref" if 'real' in filename.lower() else 'in'
            _filename = _filename.replace("_Real", "")
            _filename = _filename.replace("_mean", "")
            os.rename(os.path.join(output_dir, "Valid", filename),
                      os.path.join(output_dir, "Valid", partition, _filename))


def download_BSDS_grayscale(output_dir="./tmp/BSDS500/"):
    """Downloads the Berkeley Segmentation Dataset images, which can later be used for image denoising.

    Notes
    -----
    Training images are downloaded from `Github
    <https://github.com/aGIToz/KerasDnCNN>`_. (BSD400)
    Validation images are downloaded from `Github
    <https://github.com/cszn/DnCNN/tree/master/testsets/BSD68>`_. (BSD68)
    Validation images are further cropped to 256 x 256.

    Parameters
    ----------
    output_dir : str
        String to the directory where images will be saved.
    """
    # 180 x 180 patches.
    try:
        download_from_github(repo_addr="aGIToz/KerasDnCNN",
                             repo_dir="genData",
                             output_dir=os.path.join(output_dir, "Train", "ref"))
    except FileExistsError as err:
        pass
    # 256 x 256 patches cropped from BSD68
    try:
        download_from_github(repo_addr="cszn/DnCNN",
                             repo_dir="testsets/BSD68",
                             output_dir=os.path.join(output_dir, "Valid", "ref"))
    except FileExistsError as err:
        return
    # Crops images after downloading
    filepaths = [os.path.join(output_dir, "Valid", "ref", filename) for filename in
                 os.listdir(os.path.join(output_dir, "Valid", "ref"))]
    for filepath in tqdm(filepaths, ascii=True, desc="Cropping images from {}".format(os.path.join(output_dir, "Valid",
                                                                                                   "ref"))):
        img = imread(filepath)
        h, w = img.shape[:2]
        assert h > 256 and w > 256, "Expected h = {} > {} and w = {} > {}".format(h, 256, w, 256)
        i = np.random.randint(0, h - 256)
        j = np.random.randint(0, w - 256)
        img = img[i: i + 256, j: j + 256]
        imwrite(filepath, img)


def download_dncnn_testsets(output_dir="./tmp/testsets", testset="BSD68"):
    """Downloads one of the test sets of DnCNN from the `Paper Github
    <https://github.com/cszn/DnCNN/tree/master/testsets>`_.

    Parameters
    ----------
    output_dir : str
        String containing the path to the directory where each testset will be downloaded.
    testset : str
        One among 'BSD68' (grayscale), 'LIVE1' (RGB), 'Set12' (grayscale), 'Set14' (grayscale), 'Set5' (grayscale)
        and 'classic5' (grayscale).
    """
    download_from_github(repo_addr="cszn/DnCNN",
                         repo_dir="testsets/{}".format(testset),
                         output_dir=os.path.join(output_dir, testset, "ref"))
