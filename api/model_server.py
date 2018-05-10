from keras.models import model_from_json
import numpy as np
import settings
import utils
import redis
import time
import json

db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)

def load_model():
    json_file = open('trained_model/trained_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("trained_model/trained_model.h5")
    return model

def decode_predictions(predictions, top=3):
    preds = []
    batch_preds = []
    for pred in predictions:
        indexes = np.argpartition(pred, -top)[-top:]
        indexes = indexes[np.argsort(-pred[indexes])]
        for i in xrange(top):
            preds.append((indexes[i], pred[indexes[i]]))
        batch_preds.append(preds)
        preds = []
    return batch_preds

def predict_process():
    model = load_model()
    while True:
        queue = db.lrange(settings.IMAGE_QUEUE, 0, settings.BATCH_SIZE -1)
        image_IDs = []
        batch = None
        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = utils.b64_decoding(q["image"], (settings.IMAGE_SHAPE,))
            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])
            image_IDs.append(q["id"])
        if len(image_IDs):
            preds = model.predict(batch)
            results = decode_predictions(preds)
            for (image_id, result_set) in zip (image_IDs, results):
                output = []
                for (label, prob) in result_set:
                    res = {"label": str(label), "probability": float(prob)}
                    output.append(res)
                db.set(image_id, json.dumps(output))
            db.ltrim(settings.IMAGE_QUEUE, len(image_IDs), -1)
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    predict_process()
