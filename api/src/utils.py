import numpy as np
import base64

def b64_encoding(array):
    return base64.b64encode(array).decode("utf-8")

def b64_decoding(enc_array, shape):
    return np.frombuffer(base64.b64decode(enc_array)).reshape(shape)[np.newaxis]
