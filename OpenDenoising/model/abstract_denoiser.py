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


from abc import ABC, abstractmethod


class AbstractDenoiser(ABC):
    """Common interface for Denoiser classes. This class defines the basic functionalities that a Denoiser needs to
    have, such as __call__ function, which takes as input a noised image, and returns its reconstruction.

    Attributes
    ----------
    model_name : string
        Model string identifier.

    """

    def __init__(self, model_name="DenoisingModel"):
        self.model_name = model_name
        super().__init__()

    def __str__(self):
        """Returns denoising model name."""
        return self.model_name

    @abstractmethod
    def __call__(self, image):
        """Denoises a given image.

        Parameters
        ----------
        image : :class:`numpy.ndarray`
            Batch of noisy images. Expected a 4D array with shape (batch_size, height, width, channels).
        Returns
        -------

        """
        pass
