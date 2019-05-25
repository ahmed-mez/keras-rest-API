from numpy import frombuffer, newaxis
from base64 import b64encode, b64decode
from PIL import Image, ImageFilter
from io import BytesIO
from numpy import array, newaxis
from config import IMAGE_WIDTH, IMAGE_HEIGHT


def b64_encoding(array):
    return b64encode(array).decode("utf-8")

def b64_decoding(enc_array, shape):
    return frombuffer(b64decode(enc_array)).reshape(shape)[newaxis]

def prepare_image(image, target_width=IMAGE_WIDTH, target_height=IMAGE_HEIGHT):
    """ Prepare image to be processed
    Convert image and adapt its size to be processed by the OCR model.

    Arguments:
        image: file -- image to be prepared and processed by the api

    Returns:
        Numpy array
    """
    im = Image.open(BytesIO(image)).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (target_width, target_height), (255))
    if width > height:
        nheight = int(round((20.0 / width * height), 0))
        if (nheight == 0):
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((target_height - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((target_width - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))
    tv = list(newImage.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return array(tva)[newaxis].copy(order="C")
