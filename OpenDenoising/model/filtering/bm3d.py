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


import numpy as np

try:
    import matlab.engine
    MATLAB_IMPORTED = True
except ImportError as err:
    module_logger.warning("Matlab engine was not installed correctly. Take a look on the documentation's tutorial for \
                           its installation.")
    err_import = err
    MATLAB_IMPORTED = False

try:
    eng = matlab.engine.start_matlab()
    MATLAB_LAUNCHED = True
except matlab.engine.EngineError as err:
    err_launch = err
    MATLAB_LAUNCHED = False


def BM3D(z, sigma=25.0, profile="np"):
    """This function wraps MATLAB's BM3D [1]_ implementation available on `author's website
    <http://www.cs.tut.fi/~foi/GCF-BM3D/>`_.

    Notes
    -----
    In order to be able to use their code, you need to download it
    from the website and add the root folder to Matlab's path through "set_path". By using this function,
    and consequently their code, you are agreeing with the `license term
    <http://www.cs.tut.fi/~foi/GCF-BM3D/legal_notice.html>`_.

    Parameters
    ----------
    z : :class:`numpy.ndarray`
        4D batch of noised images. It has shape: (batch_size, height, width, channels).
    sigma : float
        Level of gaussian noise.
    profile : str
        One between {'np', 'lc', 'high', 'vn', 'vn_old'}. Algorithm's profile.

        Available for grayscale:

        * 'np': Normal profile.
        * 'lc': Fast profile.
        * 'high': High quality profile.
        * 'vn': High noise profile (sigma > 40.0)
        * 'vn_old': old 'vn' profile. Yields inferior results than 'vn'.

        Available for RGB:

        * 'np': Normal profile.
        * 'lc': Fast profile.

    Returns
    -------
    y_est : :class:`numpy.ndarray`
        4D batch of denoised images. It has shape: (batch_size, height, width, channels).

    References
    ----------
    .. [1] Dabov K, Foi A, Katkovnik V, Egiazarian K. Image denoising by sparse 3-D transform-domain collaborative
           filtering. IEEE Transactions on image processing. 2007
    """
    global MATLAB_IMPORTED, MATLAB_LAUNCHED
    assert MATLAB_IMPORTED, "Got expcetion '{}' while importing matlab.engine. Check Matlab's Engine installation.".format(err_import)
    global MATLAB_LAUNCHED
    assert MATLAB_LAUNCHED, "Got expcetion '{}' while launching matlab.engine. Check Matlab's Engine installation.".format(err_launch)
    _z = z.copy()
    rgb = True if _z.shape[-1] == 3 else False
    if rgb:
        assert (profile in ["np", "lc", "high", "vn", "vn_old"]), "Expected profile to be 'np', 'lc', 'high', 'vn' " \
                                                                  "or 'vn_old' but got {}.".format(profile)
    else:
        assert (profile in ["np", "lc"]), "Expected profile to be 'np', 'lc' bug got {}".format(profile)

    # Convert input arrays to matlab
    m_sigma = matlab.double([sigma])
    m_show = matlab.int32([0])

    if not rgb and eng.which('BM3D') == '':
        raise ModuleNotFoundError("BM3D Filter is not installed. Check your Matlab path.")
    if rgb and eng.which('CBM3D') == '':
        raise ModuleNotFoundError("CBM3D Filter is not installed. Check your Matlab path.")

    # Call BM3D function on matlab
    y_est = []
    for i in range(len(_z)):
        m_z = matlab.double(_z[i, :, :, :].tolist())
        if rgb:
            _, y = eng.CBM3D(m_z, m_z, m_sigma, profile, m_show, nargout=2)
        else:
            _, y = eng.BM3D(m_z, m_z, m_sigma, profile, m_show, nargout=2)
        y_est.append(np.asarray(y))

    y_est = np.asarray(y_est).reshape(z.shape)
    return y_est
