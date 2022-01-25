import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_model', type=str, help='input model path(pb, onnx, h5)', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-w', '--max_workspace_size', type=int, default=30)
    parser.add_argument('-t', '--trtexec_path', type=str, help='trtexec path in trt/bin',
                        default='/usr/src/tensorrt/bin/trtexec')
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input_model)
    batch_size = args.batch_size
    max_workspace_size = args.max_workspace_size
    trtexec_path = os.path.expanduser(args.trtexec_path)

    input_ext = os.path.splitext(input_path)[1]
    if input_ext == '.pb':
        from tools.convert_module import convert_pb2trt
        output_path = os.path.expanduser('../runtime_data/detect-smarteye-helmet-trt-3.engine')
        convert_pb2trt.ssd2trt(input_path, output_path, max_workspace_size, trtexec_path)
    elif input_ext == '.uff':
        from tools.convert_module import convert_uff2trt
        output_path = os.path.expanduser('../runtime_data/detect-smarteye-helmet-trt-3.engine')
        convert_uff2trt.uff2trt(input_path, output_path, max_workspace_size, trtexec_path)
    elif input_ext == '.onnx':
        from tools.convert_module import convert_onnx2trt
        output_path = os.path.expanduser('../runtime_data/detect-smarteye-helmet-trt-2.engine')
        convert_onnx2trt.onnx2trt(input_path, output_path, batch_size, max_workspace_size)

if __name__ == '__main__':
    main()
