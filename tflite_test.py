from PIL.Image import Resampling
from PIL import Image
import cv2
import numpy as np

import tensorflow as tf

if __name__ == '__main__':
    interpreter = tf.lite.Interpreter(model_path='saved_model/test4_float32.tflite')
    interpreter.allocate_tensors()

    image = Image.open("SHIQ/SpecularHighlightTest/high_light/test2.png")
    image = image.resize((256, 256))
    image_re = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("test13.jpg", image_re)
    image = (np.array(image_re)).astype(np.float32) / 255
    input_data = np.expand_dims(image, axis=0)

    print(input_data.shape)
    print(input_data)

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data0 = (interpreter.get_tensor(output_details[0]['index']) * 255).astype(np.uint8)
    output_data1 = (interpreter.get_tensor(output_details[1]['index']) * 255).astype(np.uint8)
    output_data2 = (interpreter.get_tensor(output_details[2]['index']) * 255).astype(np.uint8)

    # print(type(output_data))
    # print(output_data.transpose(0, 2, 3, 1)[0].shape)

    cv2.imwrite("14093_A_test2.jpg", output_data0[0])
    # cv2.imwrite("test10.png", output_data1[0])
    # cv2.imwrite("test11.png", output_data2[0])
