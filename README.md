# Deep Learning API using Flask, Keras, Redis, nginx and Docker
A scalable Flask API to interact with a pre-trained Keras model.

### Overview
The API uses Redis for queuing requests, batch them and feed them to the model to predict the classes then responds the client with a JSON containing the result of his request (classes with top probabilities).
In order to support heavy load and to avoid multiple model syndrome, the model load and prediction, and the receiving/sending requests run independently of each other on different processes.


Please note that the model used in this project (which is a simple digit recognition OCR model) is just an example, as the main purpose of the project is the development and deployment of the API.

### Deployment
The setup consists of 3 containers:
1. Flask app with uWSGI
2. Redis
3. nginx

All 3 containers are based respectively on the official docker images of python:2.7.15, Redis and nginx.


#### Setup
We just need to build the images and run them using `docker-compose`:

```
$ docker-compose build
$ docker-compose up
```

A head over `http://localhost:8080` should show the following message on the web page:
`Welcome to the digits OCR Keras REST API`


#### Example
We can try to submit `POST` requests to the API on the `/predict` entry point:

`$ curl -X POST -F image=@4.jpg 'http://localhost:8080/predict'`

The API response :

```
{
	"predictions": [
		{
			"label": "4",
			"probability": 0.9986100196838379
		},
		{
			"label": "9",
			"probability": 0.001174862147308886
		},
		{
			"label": "7",
			"probability": 0.00009050272637978196
		}
	],
	"success": true
}
```
### TODO
- [ ] Tests
