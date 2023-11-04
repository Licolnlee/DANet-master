import onnx
from onnx_tf.backend import prepare
import os


def onnx2pb(onnx_input_path, pb_output_path):
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model, device='cpu')  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model


if __name__ == "__main__":
    os.makedirs("tensorflow", exist_ok=True)
    onnx_input_path = 'test4.onnx'
    pb_output_path = './tensorflow/model.pb'
    TF_PATH = "tf_model"

    onnx2pb(onnx_input_path, TF_PATH)
