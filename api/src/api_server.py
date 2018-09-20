from config import (REDIS_HOST, REDIS_PORT, REDIS_DB,
                      IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_QUEUE,
                      CONSUMER_SLEEP)
from PIL import Image, ImageFilter
from utils import b64_encoding
from redis import StrictRedis
import numpy as np
import flask
import uuid
import time
import json
import io

app = flask.Flask(__name__)
db = StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def prepare_image(image):
    """ Prepare image to be processed
    Convert image and adapt its size to be processed by the OCR model.

    Arguments:
        image: file -- image to be prepared and processed by the api

    Returns:
        Numpy array
    """
    im = Image.open(io.BytesIO(image)).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), (255))
    if width > height:
        nheight = int(round((20.0 / width * height), 0))
        if (nheight == 0):
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((IMAGE_HEIGHT - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((IMAGE_WIDTH - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))
    tv = list(newImage.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return np.array(tva)[np.newaxis].copy(order="C")

@app.route("/")
def index():
    return "Welcome to the digits OCR Keras REST API"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            try:
                image = prepare_image(image)
            except Exception:
                raise
            im_id = str(uuid.uuid4())
            im_dict = {"im_id": im_id, "image": b64_encoding(image)}
            # send image to the redis queue
            db.rpush(IMAGE_QUEUE, json.dumps(im_dict))
            while True:
                # start polling
                output = db.get(im_id)
                if output is not None:
                    # image processed, try to get predictions
                    try:
                        output = output.decode("utf-8")
                        data["predictions"] = json.loads(output)
                        db.delete(im_id)
                    except Exception:
                        raise
                    break
                time.sleep(CLIENT_SLEEP)
            data["success"] = True
            return flask.jsonify(data), 200
    return flask.jsonify(data), 400

if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)
