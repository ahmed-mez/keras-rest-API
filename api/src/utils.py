from numpy import frombuffer, newaxis
from base64 import b64encode, b64decode

def b64_encoding(array):
    return b64encode(array).decode("utf-8")

def b64_decoding(enc_array, shape):
    return frombuffer(b64decode(enc_array)).reshape(shape)[newaxis]
