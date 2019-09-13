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
import onnx
import tf2onnx
import logging
import tensorflow as tf


def pb2onnx(path_to_pb, path_to_onnx, input_node_names=["input"], output_node_names=["output"]):
    """Converts Tensorflow's ProtoBuf file to Onnx model.

    Parameters
    ----------
    path_to_pb : str
        String containing the path to the .pb file containing the tensorflow graph to convert to onnx.
    path_to_onnx : str
        String containing the path to the location where the .onnx file will be saved.
    input_node_names : list
        List of strings containing the input node names
    output_node_names : list
        List of strings containing the output node names
    """
    nodes = [n for n in tf.get_default_graph().as_graph_def().node]
    assert len(nodes) == 0, "Expected Tensorflow computational graph to be empty. {}".format(len(nodes))
    # Parsing pb file
    with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.graph_util.import_graph_def(graph_def)
    graph = tf.get_default_graph()

    # Creating onnx graph
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph, input_node_names, output_node_names)
    model_proto = onnx_graph.make_model("test")
    with open(path_to_onnx, "wb") as f:
        f.write(model_proto.SerializeToString())


def freeze_tf_graph(model_file=None, output_filepath='./output_graph.pb', output_node_names=None):
    """Freezes tensorflow graph, then writes it in a new .pb file.

    Parameters
    ----------
    model_file : str
        String to the checkpoint or saved_model file containing the target graph. If None, assumes that the graph was
        already loaded.
    output_filename : str
        Path of the output file.
    output_node_names : str
        Name of the identifier of output nodes.
    """
    if model_file is not None:
        nodes = [n for n in tf.get_default_graph().as_graph_def().node]
        assert len(nodes) == 0, "You are trying to freeze the Tensorflow graph on file {}," \
                                " but the current tensorflow session is not empty. Found {} nodes" \
                                " in the graph.".format(model_file, len(nodes))
        filename, ext = os.path.splitext(model_file)
        if ext == '.meta':
            session = tf.Session()
            saver = tf.train.import_meta_graph(model_file)
            saver.restore(session, tf.train.latest_checkpoint(os.path.dirname(model_file)))
        elif ext == '.pb':
            session = tf.Session()
            tf.saved_model.load(session, ['serve'], os.path.dirname(model_file))
        else:
            raise ValueError("Expected extension to be .meta or .pb, but got {}".format(ext))
    else:
        session = tf.get_default_session()

    if output_node_names is None:
        logging.warning("Output node names not specified. Falling into automatic mode (name is 'output').")
        logging.warning("Name of nodes present in graph:")
        for n in tf.get_default_graph().as_graph_def().node:
            logging.debug("[Node] {}".format(n.name))
        output_node_names = ['output']

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(session, session.graph_def, output_node_names)
    with open(output_filepath, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())


def onnx_dynamic_shapes(input_filepath, output_filepath, channels_first=False):
    """Converts inputs and outputs static shapes of an Onnx graph to dynamic shapes.

    Parameters
    ----------
    input_filepath : str
        String containing the path to the .onnx file you want to convert
    output_filepath : str
        String containing the path for the new onnx model to be saved.
    channels_first : bool
        If True, interprets input shapes as NCHW. If False, interprets input shapes as NHWC.

    """
    model = onnx.load(input_filepath)
    ndims = len(model.graph.input[-1].type.tensor_type.shape.dim)

    assert ndims == 4, "Expected input tensor to have 4D, but got {}D".format(ndims)

    # Determines whether NCHW or NHWC
    h = 2 if channels_first else 1
    w = 3 if channels_first else 2

    for i in range(len(model.graph.input)):
        if "input" in model.graph.input[i].name.lower():
            break

    for j in range(len(model.graph.output)):
        if "output" in model.graph.output[j].name.lower():
            break

    # Batch dynamic dimension
    model.graph.input[i].type.tensor_type.shape.dim[0].dim_param = "?"
    # Height dynamic dimension
    model.graph.input[i].type.tensor_type.shape.dim[h].dim_param = "?"
    # Width dynamic dimension
    model.graph.input[i].type.tensor_type.shape.dim[w].dim_param = "?"

    # Batch dynamic dimension
    model.graph.output[j].type.tensor_type.shape.dim[0].dim_param = "?"
    # Height dynamic dimension
    model.graph.output[j].type.tensor_type.shape.dim[h].dim_param = "?"
    # Width dynamic dimension
    model.graph.output[j].type.tensor_type.shape.dim[w].dim_param = "?"

    onnx.save(model, output_filepath)
