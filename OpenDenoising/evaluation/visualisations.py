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
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")


class Visualisation:
    """Wraps visualisation functions.

    Attributes
    ----------
    func : function
        Reference to a function that will create the plot.
    name : str
        Visualisation's name.
    """
    def __init__(self, func, name):
        self.func = func
        self.name = name

    def __call__(self, file_dir, **kwargs):
        """

        Parameters
        ----------
        file_dir : str
            Path to the directory containing the files that will be used for constructing the visualisation.
        kwargs : dict
            Keyword arguments. Used for passing optional arguments for the visualisation.

        Examples
        --------
        Assuming you are output your benchmark results to "./results", and that you have a Benchmark with name
        "MyBenchTests", you can use the boxplot function to generate visualisations from the output .csv files.

        >>>from OpenDenoising.evaluation import Visualisation, boxplot
        >>>vis = Visualisation(func=boxplot, name="boxplot_PSNR")
        >>>vis(file_dir="./results/")
        """
        self.func(file_dir, **kwargs)

    def __str__(self):
        return self.name


def boxplot(csv_dir, output_dir, metric="PSNR", show=True):
    """Wraps Seaborn boxplot function.

    Parameters
    ----------
    csv_dir : str
        String containing the path to CSV file directory holding the data.
    output_dir : str
        String containing the path to save the image.
    metric : str
        String containing the name of the metric being shown by the plot.
    show : bool
        If bool is True, shows the plot rather than only saving it to output_path.
    """
    dataframe = pd.read_csv(os.path.join(csv_dir, "Partial_Results.csv"))
    n_models = dataframe.nunique()['Model']
    sns.set(rc={'figure.figsize': (n_models * 5, 10)})
    sns.boxplot(x="Model", y=metric, hue="Dataset", data=dataframe).set_title(metric)
    plt.savefig(os.path.join(output_dir, "boxplot_{}.png".format(metric)))
    if show:
        plt.show()


def circles_plot(csv_path, output_path="./results/Figures", metrics=None, show=True):
    if metrics is None:
        metrics = ["Model", "PSNR Mean", "SSIM Mean"]

    dataframe = pd.read_csv(os.path.join(csv_path, "General_Results.csv"))
    dataframe = dataframe[metrics]
    df_matrix = dataframe.values
    train_info_path = os.path.join(csv_path, "TrainingInfo")
    train_filenames = os.listdir(train_info_path)
    train_infos = [os.path.join(train_info_path, train_info) for train_info in train_filenames]
    train_data = {}

    for train_info, train_filename in zip(train_infos, train_filenames):
        with open(train_info, "r") as f:
            data = json.load(f)
            model_name = train_filename.split("_info.json")[0]
            train_data[model_name] = data["NumParams"]

    plt.figure()
    max_params = max([train_data[key] for key in train_data])

    for model in train_data:
        metrics = df_matrix[np.where(df_matrix[:, 0] == model)][0]
        plt.scatter([metrics[2]], [metrics[1]], s=(500 * train_data[model]) // max_params, label=model)
    plt.ylabel("PSNR (dB)")
    plt.xlabel("SSIM")
    plt.legend()
