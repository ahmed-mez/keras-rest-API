from PIL import Image, ImageFilter
import numpy as np
import settings
import utils
import flask
import redis
import uuid
import time
import json
import io

app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
    port=settings.REDIS_PORT, db=settings.REDIS_DB)

def prepare_image(image):
    IMAGE_WIDTH = settings.IMAGE_WIDTH
    IMAGE_HEIGHT = settings.IMAGE_HEIGHT
    im = image.convert('L')
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
    return np.array(tva)[np.newaxis]

@app.route("/")
def index():
    return "Welcome to the digits OCR Keras REST API"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image)
            image = image.copy(order="C")
            im_id = str(uuid.uuid4())
            im_dict = {"id": im_id, "image": utils.b64_encoding(image)}
            db.rpush(settings.IMAGE_QUEUE, json.dumps(im_dict))
            while True:
                output = db.get(im_id)
                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(im_id)
                    break
                time.sleep(settings.CLIENT_SLEEP)
            data["success"] = True
    return flask.jsonify(data)

if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)
