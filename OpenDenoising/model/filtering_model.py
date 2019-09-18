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


from OpenDenoising.model import AbstractDenoiser


class FilteringModel(AbstractDenoiser):
    """FilteringModel represents a Denoising Model that does not depend on Neural Networks.

    Attributes
    ----------
    model_function : :class:`function`
        Filtering denoising function. It should accept at least one argument, a image batch :class:`numpy.ndarray`.
        It should also have only one return, another :class:`numpy.ndarray` with same shape, corresponding to the
        denoising result.
    """

    def __init__(self, model_name="FilteringModel"):
        self.model_function = None
        super().__init__(model_name=model_name)

    def charge_model(self, model_function, **kwargs):
        """Charges the denoising function into the class wrapper.

        Parameters
        ----------
        model_function : :class:`function`
            Filtering denoising function. It should accept at least one argument, a image batch :class:`numpy.ndarray`.
            It should also have only one return, another :class:`numpy.ndarray` with same shape, corresponding to the
            denoising result. Notice that, if your function needs more arguments than the noisy image batch, these can
            be passed through keyword arguments to charge_model (see examples section).
        """
        def partial_func(image):
            return model_function(image, **kwargs)
        self.model_function = partial_func

    def __call__(self, image):
        """Denoises a batch of images noised_image

        Parameters
        ----------
        image : :class:`numpy.ndarray`
            batch of images with shape (batch_size, height, width, channels)

        Returns
        -------
        :class:`numpy.ndarray`
            batch of images denoised by model_func, with shape (b, h ,w , c)
        """
        denoised_image = self.model_function(image)
        return denoised_image

    def __repr__(self):
        return "Filtering model name: {}".format(self)

    def __len__(self):
        return 1
