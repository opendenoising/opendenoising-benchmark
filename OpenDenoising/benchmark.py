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
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from OpenDenoising import data, model, evaluation


def print_denoising_results(image_noised, image_clean, image_denoised, denoiser_name, filename, output_dir):
    """Creates an image comparing three images: noised, clean and denoised.

    image_noised : :class:`numpy.ndarray`
        2D or 3D image array corresponding to the image corrupted by noise.
    image_clean : :class:`numpy.ndarray`
        2D or 3D image array corresponding to the original, clean image.
    image_denoised : :class:`numpy.ndarray`
        2D or 3D image array corresponding to the neural network prediction.
    filename : str
        Name of the file being denoised.
    output_dir : str
        String containing the path to the output directory.

    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Shows noisy images
    axes[0].imshow(np.squeeze(image_noised), cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Noised Image")

    # Shows clean images
    axes[1].imshow(np.squeeze(image_clean), cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Clean Image")

    # Shows restored images
    axes[2].imshow(np.squeeze(image_denoised), cmap="gray")
    axes[2].axis("off")
    axes[2].set_title("Restored Image")

    plt.suptitle("Denoising results on {}".format(filename))

    plt.savefig(os.path.join(output_dir, "RestoredImages", denoiser_name, filename))
    plt.close("all")


class Benchmark:
    """Benchmark class.

    The purpose of this class is to evaluate models on given datasets. The evaluations are registered through the
    function "register_function". After registering the desired metrics, you can run numeric evaluations through
    "numeric_evaluation". To further visualize the results, you may also register visualization functions (to generate
    figures) through "register_visualization". Once you have computed the metrics, you can call "graphic_evaluation"
    to generate the registered plots.

    Attributes
    ----------
    models : list
        List of :class:`model.AbstractDenoiser` objects.
    metrics : list
        List of dictionaries holding the following fields,

        * Name (str): metric name.
        * Func (:class:`function`): function object.
        * Value (list): List of values, one for each file in dataset.
        * Mean (float): Mean of "Values" field.
        * Variance (float): Variance of "Values" field.
    datasets : list
        List of :class:`data.AbstractDatasetGenerator`
    visualizations : list
        List of visualization functions.
    name : str
        String holding the identifier of evaluations being performed.
    output_dir : str
        String holding the path to the output directory.
    partial : :class:`pandas.DataFrame`
        DataFrame holding per-file denoising results.
    general : :class:`pandas.DataFrame`
        DataFrame holding aggregates of denoising results.
    """

    def __init__(self, name="evaluation", output_dir="./results"):
        # Lists
        self.models = []
        self.metrics = [{"Name": "Evaluation Time", "Func": None, "Value": [], "Mean": -1.0, "Variance": -1.0}]
        self.datasets = []
        self.visualizations = []

        # Evaluator name
        self.name = name

        # Output directory
        self.output_dir = os.path.join(output_dir, name)

        # Create non-existent dirs
        if not os.path.isdir(os.path.join(self.output_dir, "RestoredImages")):
            os.makedirs(os.path.join(self.output_dir, "RestoredImages"))
        if not os.path.isdir(os.path.join(self.output_dir, "Visualizations")):
            os.makedirs(os.path.join(self.output_dir, "Visualizations"))

        # Pandas DataFrames
        self.partial = pd.DataFrame(columns=["Model", "Dataset", "Filename"])
        self.general = pd.DataFrame(columns=["Model", "Dataset", "Number of Parameters"])

    def __register_models(self, model_list):
        for mdl in model_list:
            self.models.append(mdl)
            if not os.path.isdir(os.path.join(self.output_dir, "RestoredImages", str(mdl))):
                os.makedirs(os.path.join(self.output_dir, "RestoredImages", str(mdl)))

    def __register_datasets(self, dataset_list):
        for dataset in dataset_list:
            self.datasets.append(dataset)

    def __register_metrics(self, metric_list):
        for metric in metric_list:
            assert metric.np_metric is not None, "Expected metric {} to have a metric function defined on" \
                                                 "numpy arrays.".format(metric)
            self.metrics.append({
                "Name": str(metric),
                "Func": metric,
                "Value": [],
                "Mean": -1.0,
                "Variance": -1.0
            })
        for metric in self.metrics:
            # Adds metric column on partial dataframe
            self.partial[metric["Name"]] = None
            self.partial[metric["Name"]] = None
            # Adds metric column on general dataframe
            self.general[metric["Name"] + " Mean"] = None
            self.general[metric["Name"] + " Variance"] = None

    def __register_visualizations(self, vis_list):
        for vis in vis_list:
            self.visualizations.append({
                "Name": str(vis),
                "Func": vis
            })

    def register(self, obj_list):
        """Register datasets, models, metrics or evaluations into Benchmark object.

        Parameters
        ----------
        obj_list : list
            List of instances of datasets, models, metrics or evaluations to be registered to internal lists.
        """
        for obj in obj_list:
            if isinstance(obj, model.AbstractDenoiser):
                # obj is a model
                self.__register_models([obj])
            elif isinstance(obj, data.AbstractDatasetGenerator):
                # Obj is a dataset
                self.__register_datasets([obj])
            elif isinstance(obj, evaluation.Metric):
                # Obj is metric
                self.__register_metrics([obj])
            elif isinstance(obj, evaluation.Visualisation):
                # Obj is visualisation
                self.__register_visualizations([obj])

    def evaluate_model_on_dataset(self, denoiser, test_generator):
        """Evaluates denoiser on dataset represented by test_generator.

        Parameters
        ----------
        denoiser : :class:`model.AbstractDeepLearningModel`
            Denoiser object
        test_generator : :class:`data.AbstractDatasetGenerator`
            Dataset generator object. It generates data to evaluate the denoiser.

        Returns
        -------
        list
            List of evaluated metrics.
        """
        for metric in self.metrics:
            # Reset values
            metric["Value"] = []
            metric["Mean"] = -1.0
            metric["Variance"] = -1.0

        pgbar = tqdm(range(len(test_generator)))
        for i in pgbar:
            pgbar.set_description("Inferencing on image {}".format(test_generator.filenames[i]))
            # Draws batch from test_generator
            image_noised, image_clean = test_generator[i]
            # Denoises image_noised
            start = time.time()
            image_denoised = denoiser(image_noised)
            finish = time.time()

            print_denoising_results(image_noised, image_clean, image_denoised, str(denoiser),
                                    str(test_generator) + "_" + test_generator.filenames[i],
                                    self.output_dir)

            self.metrics[0]["Value"].append(finish - start)
            for metric in self.metrics[1:]:
                # Evaluates each metric registerd in metrics.
                metric["Value"].append(
                    metric["Func"](np.squeeze(image_clean), np.squeeze(image_denoised))
                )
        values = {metric["Name"]: metric["Value"] for metric in self.metrics}
        # Calculating aggregate metrics
        for metric in self.metrics:
            metric["Mean"] = np.mean(metric["Value"])
            metric["Variance"] = np.var(metric["Value"])

        # Constructing row to write on csv file
        row = [str(denoiser), str(test_generator), len(denoiser)]
        for metric in self.metrics:
            row.extend([metric["Mean"], metric["Variance"]])

        return values, row

    def evaluate(self):
        """Perform the entire evaluation on datasets and models.

        For each pair (model, dataset), runs inference on each dataset image using model. The results are stored
        in a pandas DataFrame, and later written into two csv files:

        1. (EVALUATION_NAME)/General_Results: contains aggregates (mean and variance of each metric) about the ran tests
        2. (EVALUATION_NAME)/Partial_Results: contains the performance of each model on each dataset image.

        These tables are stored on 'output_dir'. Moreover, the visual restoration results are stored into
         'output_dir/EVALUATION_NAME/RestoredImages' folder. If you have visualisations registered into your evaluator,
         the plots are saved in 'output_dir/EVALUATION_NAME/Figures' folder.
        """
        if not self.models:
            raise IndexError("Empty models list.")
        if not self.datasets:
            raise IndexError("Empty dataset list.")
        if not self.metrics:
            raise IndexError("Empty metrics list.")
        values_dict = {metric["Name"]: [] for metric in self.metrics}
        model_labels = []
        dataset_labels = []
        filenames = []
        for i, dataset in enumerate(self.datasets):
            # Iterates over datasets
            values_list = []
            for mdl in self.models:
                # Iterates over models
                print("[Benchmark] Evaluating model {} on dataset {}".format(str(mdl), str(dataset)))
                filenames.extend(dataset.filenames)
                dataset_labels.extend([str(dataset)] * len(dataset))
                model_labels.extend([str(mdl)] * len(dataset))
                # Get values and metric statistics (mean and variance)
                model_values, metric_statistics = self.evaluate_model_on_dataset(mdl, dataset)
                dict_row = {
                    column: metric for column, metric in zip(self.general.columns, metric_statistics)
                }
                self.general = self.general.append(dict_row, ignore_index=True)
                values_list.append(model_values)
            for model_values in values_list:
                for key in model_values:
                    # Appends model results to dictionary
                    values_dict[key].extend(model_values[key])
        self.partial["Model"] = model_labels
        self.partial["Dataset"] = dataset_labels
        self.partial["Filename"] = filenames
        for key in values_dict:
            self.partial[key] = values_dict[key]
        self.partial.to_csv(os.path.join(self.output_dir, "Partial_Results.csv"))
        self.general.to_csv(os.path.join(self.output_dir, "General_Results.csv"))
        self.visualize(csv_dir=self.output_dir, output_dir=os.path.join(self.output_dir, "Visualizations"))

    def visualize(self, csv_dir, output_dir):
        """Plot visualizations. """
        for visualization in self.visualizations:
            visualization["Func"](csv_dir, output_dir=output_dir)

    def __str__(self):
        output_string = "Registered metrics:\n"
        for metric in self.metrics:
            output_string = output_string + "Metric: {}\tMean: {}\tVariance: {}\n".format(metric["Name"],
                                                                                          metric["Mean"],
                                                                                          metric["Variance"])
        return output_string
