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
from OpenDenoising.data import BlindDatasetGenerator, FullDatasetGenerator, CleanDatasetGenerator


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def create(path, batch_size=32, n_channels=1, preprocessing=None, name="Dataset", noise_config=None):
        """Creates a BlindDatasetGenerator, CleanDatasetGenerator or FullDatasetGenerator

        Parameters
        ----------
        path : str
            String containing the path to the directory where in/ref folders are located.
        batch_size : int
            Size of image batches.
        preprocessing : list
            List of preprocessing functions, which will be applied to each image.
        name : str
            String containing the dataset's name.
        noise_config : dict
            Dictionary whose keys are corruption functions (see :mod:`data.utils`) and the value
            corresponds to a list of function arguments.
        """
        dirs = [folder for folder in os.listdir(path) if os.path.isdir(path)]

        assert ('in' in dirs or 'ref' in dirs), "You should provide noisy inputs in " \
                                                "{} or references in {}".format(os.path.join(path, "in"),
                                                                                os.path.join(path, "ref"))
        if 'in' in dirs and 'ref' not in dirs:
            # Blind dataset
            return BlindDatasetGenerator(path=path, batch_size=batch_size, n_channels=n_channels, shuffle=True,
                                         name=name, preprocessing=preprocessing)
        elif 'in' in dirs and 'ref' in dirs:
            # Full dataset
            return FullDatasetGenerator(path=path, batch_size=batch_size, n_channels=n_channels, shuffle=True,
                                        name=name, preprocessing=preprocessing)
        elif 'in' not in dirs and 'ref' in dirs:
            # Clean dataset
            return CleanDatasetGenerator(path=path, batch_size=batch_size, n_channels=n_channels, shuffle=True,
                                         name=name, preprocessing=preprocessing, noise_config=noise_config)