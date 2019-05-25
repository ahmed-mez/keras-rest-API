from config import (REDIS_HOST, REDIS_PORT, REDIS_DB,
                      IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_QUEUE,
                      CONSUMER_SLEEP, LOG_DIR)
from flask import Flask, request, jsonify
from utils import b64_encoding, prepare_image
from redis import StrictRedis
from uuid import uuid4
import logging
import json
import time

app = Flask(__name__)
db = StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
logging.basicConfig(filename=LOG_DIR+"/api.log", level=logging.INFO)


@app.route("/")
def index():
    return "Welcome to the digits OCR Keras REST API"

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("image"):
            image = request.files["image"].read()
            try:
                image = prepare_image(image)
            except Exception:
                logging.exception("Error in preparing image")
                return jsonify(data), 400
            im_id = str(uuid4())
            im_dict = {"im_id": im_id, "image": b64_encoding(image)}
            # send image to the redis queue
            db.rpush(IMAGE_QUEUE, json.dumps(im_dict))
            logging.info("New image added to queue with id: %s", im_id)
            while True:
                # start polling
                output = db.get(im_id)
                if output is not None:
                    # image processed, try to get predictions
                    try:
                        output = output.decode("utf-8")
                        data["predictions"] = json.loads(output)
                        logging.info("Got predictions for image with id: %s", im_id)
                    except Exception:
                        logging.exception("Error in getting predictions for image with id: %s", im_id)
                        return jsonify(data), 400
                    finally:
                        db.delete(im_id)
                        logging.info("Deleting image from queue with id: %s", im_id)
                    break
                time.sleep(CONSUMER_SLEEP)
            data["success"] = True
            logging.info("Send result for image with id: %s", im_id)
            return jsonify(data), 200
    logging.warning("Invalid request with file %s", request.files)
    return jsonify(data), 400

if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)
