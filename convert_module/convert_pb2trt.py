import os
import argparse
import uff
import tensorrt as trt
import graphsurgeon as gs
import subprocess

from tools.convert_module import config_model_ssd_mobilenet_v2 as model

# initialize
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')


def pb2trt(input_path, output_path, max_workspace_size, trtexec_path):
    uff_path = os.path.splitext(input_path)[0] + '.uff'

    # compile model into TensorRT
    if not os.path.isfile(output_path):
        dynamic_graph = model.add_plugin(gs.DynamicGraph(input_path))
        uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), model.output_name, output_filename=uff_path, text=True)

        command = f"{trtexec_path} --uff={uff_path} --output=MarkOutput_0 --uffInput=Input,3,300,300 " \
                  f"--workspace={max_workspace_size} --fp16 --saveEngine={output_path}"
        subprocess.call(command, shell=True)
        print("Completed creating SSD(PB) to TRT Engine")
