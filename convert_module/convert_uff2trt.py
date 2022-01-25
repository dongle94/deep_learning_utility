import os
import argparse
import tensorrt as trt
import subprocess


# initialize
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')


def uff2trt(input_path, output_path, max_workspace_size, trtexec_path):
    # compile model into TensorRT
    if not os.path.isfile(output_path):
        command = f"{trtexec_path} --uff={input_path} --output=MarkOutput_0 --uffInput=Input,3,300,300 " \
                  f"--workspace={max_workspace_size} --fp16 --saveEngine={output_path}"
        subprocess.call(command, shell=True)
        print("Completed creating SSD(uff) to TRT Engine")
