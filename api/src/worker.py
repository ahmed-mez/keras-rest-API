from config import (IMAGE_SHAPE, IMAGE_QUEUE, BATCH_SIZE,
                      WORKER_SLEEP, REDIS_HOST, REDIS_PORT, REDIS_DB,
                      WEIGHTS_JSON, WEIGHTS_H5, LOG_DIR)
from numpy import argpartition, argsort, vstack
from keras.models import model_from_json
from utils import b64_decoding
from redis import StrictRedis
import logging
import json
import time

db = StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
logging.basicConfig(filename=LOG_DIR+"/worker.log", level=logging.INFO)


def load_model():
    """ Load the keras model in memory
    """
    try:
        json_file = open(WEIGHTS_JSON, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
    except Exception:
        raise
    model = model_from_json(loaded_model_json)
    model.load_weights(WEIGHTS_H5)
    return model

def decode_predictions(predictions, top=3):
    """ Interpret the predictions
    
    Arguments:
        predictions [[float]] -- predictions made by the model
    Yields:
        generator([float]) -- sorted result
    """
    for pred in predictions:
        indexes = argpartition(pred, -top)[-top:]
        indexes = indexes[argsort(-pred[indexes])]
        preds = list()
        for i in xrange(top):
            preds.append((indexes[i], pred[indexes[i]]))
        yield preds


def predict_process(target_shape=IMAGE_SHAPE):
    """ Worker process, load model and poll for images
    """
    model = load_model()
    assert model is not None
    logging.info("Model loaded successfully, start polling for images")
    while True:
        # start polling
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE -1)
        image_IDs = []
        batch = None
        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = b64_decoding(q["image"], (target_shape,))
            if batch is None:
                batch = image
            else:
                batch = vstack([batch, image])
            image_IDs.append(q["im_id"])
        if len(image_IDs):
            # queue contains images to be processed
            try:
                preds = model.predict(batch)
                results = decode_predictions(preds)
                logging.info("Batch predicted successfully with images ids: %s", image_IDs)
            except Exception:
                logging.exception("Error in prediction, batch with images ids: %s", image_IDs)
                raise
            for (image_id, result_set) in zip (image_IDs, results):
                output = []
                for (label, prob) in result_set:
                    res = {"label": str(label), "probability": float(prob)}
                    output.append(res)
                logging.info("Setting predictions in queue for image with id: %s", image_id)
                db.set(image_id, json.dumps(output))
            db.ltrim(IMAGE_QUEUE, len(image_IDs), -1)
        time.sleep(WORKER_SLEEP)

if __name__ == "__main__":
    logging.info("Starting process..")
    predict_process()
