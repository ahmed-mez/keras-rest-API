# initialize image dimensions
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SHAPE = IMAGE_WIDTH*IMAGE_WIDTH

# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
WORKER_SLEEP = 0.25
CONSUMER_SLEEP = 0.25

# initialize Redis connection settings
REDIS_HOST = "redis"
REDIS_PORT = 6379
REDIS_DB = 0

# weights files
WEIGHTS_JSON = "/api/trained_model/trained_model.json"
WEIGHTS_H5 = "/api/trained_model/trained_model.h5"

# logging location
LOG_DIR = "/api/logs"