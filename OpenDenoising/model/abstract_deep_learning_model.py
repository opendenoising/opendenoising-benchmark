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

from abc import abstractmethod
from OpenDenoising.model import AbstractDenoiser


class AbstractDeepLearningModel(AbstractDenoiser):
    """Common interface for Deep Learning based image denoisers."""

    def __init__(self, model_name="DeepLearningModel", logdir="./training_logs/", framework=None, return_diff=False):
        """Common interface for Deep Learning based image denoisers.

        Attributes
        ----------
        model
            Object representing the Denoiser Model in the framework used.
        logdir : str
            String containing the path to the model log directory. Such directory will contain training information,
            as well as model checkpoints.
        train_info : dict
            Dictionary containing the time spent on training, how much parameters the network has, and if it has been
            trained.
        framework : str
            String containing the name of the chosen framework (e.g. Keras, Tensorflow, Pytorch).
        return_diff : bool
            If True, return the difference between predicted image, and image.
        """
        self.model = None
        self.logdir = logdir
        self.train_info = None
        self.framework = framework
        self.return_diff = return_diff
        super().__init__(model_name)

    @abstractmethod
    def charge_model(self):
        pass

    @abstractmethod
    def train(self, train_generator, valid_generator=None):
        pass

    def __repr__(self):
        return "Model name: {}, Framework: {}".format(self, self.framework)

    @abstractmethod
    def __call__(self, image):
        pass

    @abstractmethod
    def __len__(self):
        pass
