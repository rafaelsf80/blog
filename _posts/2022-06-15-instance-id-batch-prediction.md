---
title: "Adding InstanceID on Vertex AI Batch Prediction"
description: "In Vertex AI Batch Prediction there is no way to pass Instance ID (or key ID) along with the inputs when using XGBoost or Sickit-learn prebuilt containers. This post shows how to do it using an Custom Container"
toc: true
comments: true
layout: post
categories: ["Vertex AI"]
image: images/vertex.png
author: Rafael Sanchez
---

## Summary 

In Vertex AI **Batch Prediction** [there is no way](https://issuetracker.google.com/issues/203029524) to pass Instance ID (or key ID) along with the inputs when using XGBoost or Sickit-learn prebuilt containers. Due to the parallelization done in Vertex AI, 
the order of inputs can not be maintained, so even if the inputs are printed with outputs, because they are out of order, it may be complicated for large datasets to know what prediction results map to which instances.

A feature request is open and can be tracked [here](https://issuetracker.google.com/issues/202080076).

This tutorial shows a Custom Container for predictions that solves this issue for a Scikit learn model.
Using inputs with Instance IDs (on first column), they are removed before calling the prediction. 

Example:

For this input (instance ID on first column):
```json
[1,-0.37078722949973075,-0.09383565010748596,-0.11347464767250347,0.12246106838945217,0.10186437443386016,0.1905715671716009]
```
The following output is printed in Vertex AI Batch prediction (note instance ID is converted to `float`):
```json
{"instance": [1.0,-0.37078722949973075,-0.09383565010748596,-0.11347464767250347,0.12246106838945217,0.10186437443386016,0.1905715671716009]
], "prediction": 46}

```


## Custom Container image

A Custom Container image for predictions is required. Custom Container image [requires](https://cloud.google.com/ai-platform-unified/docs/predictions/custom-container-requirements#image) that the container must run an HTTP server. 
Specifically, the container must listen and respond to liveness checks, health checks, and prediction requests.

This tutorial uses **FastAPI and Uvicorn** to implement the HTTP server. 
The HTTP server must listen for requests on `0.0.0.0`. [Uvicorn](https://www.uvicorn.org) is an ASGI web server implementation for Python. 
Uvicorn currently supports HTTP/1.1 and WebSockets. 
[Here](https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker) is a docker image with Uvicorn managed by Gunicorn for high-performance FastAPI web applications in Python 3.6+ with performance auto-tuning. 
An uvicorn server is launched with:
```bash
uvicorn main:app --host 0.0.0.0 --port 7080
```

## Set up

1. Train the model with `python3 train/train.py`. The training dataset is located at [`train/input_data.csv`](train/input_data.csv)
2. Move the resulting ..pkl model to the `custom/model` directory. The `custom` directory contains the `Dockerfile` to generate the Custom Container image. Follow ths instructions below under section **Upload and Online deployment in Vertex AI prediction**. It will upload the model to Vertex AI and, although not required for batch, will create an online endpoint.
3. Make a Batch prediction following instructions under section below **Batch prediction on Vertex AI**.


## Running docker locally

Build and run locally (for info see [here](https://cloud.google.com/ai-platform/prediction/docs/getting-started-pytorch-container#run_the_container_locally_optional))
```bash
docker build -t demo_basic_model .
docker run -p 7080:7080 -e AIP_HTTP_PORT=7080 \
    -e AIP_HEALTH_ROUTE=/health \
    -e AIP_PREDICT_ROUTE=/predict \
    demo_basic_model:latest
[...]
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7080 (Press CTRL+C to quit)
```

To access container shell:
```bash
docker run -it --rm -p 7080:7080 \
    --name=demo_basic_model \
    -e AIP_HTTP_PORT=7080 \
    -e AIP_HEALTH_ROUTE=/health \
    -e AIP_PREDICT_ROUTE=/predict \
    -e AIP_STORAGE_URI='gs://argolis-vertex-europewest4' \
    0395efe5870d \
    /bin/bash
```

Prediction example:
```bash
curl -i -X POST http://localhost:7080/predict  -d "{\"instances\": [[1,-0.37078722949973075,-0.09383565010748596,-0.11347464767250347,0.12246106838945217,0.10186437443386016,0.1905715671716009],[2,-0.37078722949973075,-0.09383565010748596,-0.11347464767250347,0.12246106838945217,0.10186437443386016,0.1905715671716009]]}"
HTTP/1.1 200 OK
date: Sun, 03 Jul 2022 14:54:39 GMT
server: uvicorn
content-length: 21
content-type: application/json

{"predictions":[5,5]}
```

Prediction example with local file `instance_test.json`:
```bash
curl -i -X POST -d @custom/instances_test.json   -H "Content-Type: application/json; charset=uost:7080/predict 
HTTP/1.1 200 OK
date: Sun, 03 Jul 2022 14:56:39 GMT
server: uvicorn
content-length: 21
content-type: application/json

{"predictions":[8,8]}
```

## Upload and Online deployment in Vertex AI prediction

Push docker image to **Artifact Registry**:
```bash
gcloud auth configure-docker europe-west4-docker.pkg.dev
gcloud builds submit --tag europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/demo_basic_model
```

Upload model to Vertex AI prediction:
```python
from google.cloud import aiplatform

STAGING_BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'

aiplatform.init(project=PROJECT_ID, staging_bucket=STAGING_BUCKET, location=LOCATION)

DEPLOY_IMAGE = 'europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/demo_basic_model' 
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORTS = [7080]

model = aiplatform.Model.upload(
    display_name=f'custom-model-uvicorn',    
    description=f'Scikit-learn mdoel with Uviron and FastAPI',
    serving_container_image_uri=DEPLOY_IMAGE,
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=SERVING_CONTAINER_PORTS,
)

print(model.resource_name)

# Retrieve a Model on Vertex
model = aiplatform.Model(model.resource_name)
print(model.resource_name)

# Deploy model
endpoint = model.deploy(
      machine_type='n1-standard-2', 
      sync=False
)
endpoint.wait()

# Retrieve an Endpoint on Vertex
endpoint = aiplatform.Endpoint('projects/989788194604/locations/europe-west4/endpoints/4614905388473516032')
print(endpoint.predict([[1, 0.16877874957321273,0.20526086240043687,0.011701852935588793,1.3426588560447468,0.28517943828011344,-0.48931828100278363],
                        [2, 0.16877874957321273,0.20526086240043687,0.011701852935588793,1.3426588560447468,0.28517943828011344,-0.48931828100278363]]))
# Output: [8 8]
```

Predict using REST API online endpoint:
```bash
curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" \
https://europe-west4-aiplatform.googleapis.com/v1alpha1/projects/989788194604/locations/europe-west4/endpoints/4614905388473516032:predict \
-d "{\"instances\": [[1,-0.37078722949973075,-0.09383565010748596,-0.11347464767250347,0.12246106838945217,0.10186437443386016,0.1905715671716009],[2,-0.37078722949973075,-0.09383565010748596,-0.11347464767250347,0.12246106838945217,0.10186437443386016,0.1905715671716009]]}"
{
  "predictions": [
    8,
    8
  ],
  "deployedModelId": "8016169842207883264",
  "model": "projects/989788194604/locations/europe-west4/models/1804377746017615872",
  "modelDisplayName": "custom-model-uvicorn"
}
```

## Batch prediction on Vertex AI

Launching a Batch prediction on Vertex AI with a table including Instance ID (`batch_input_data_with_id.csv`) on a first column generates now these outputs. 
Note results are in GCS in the folder `prediction-<model-display-name>-<job-create-time>`. Inside of it multiple files of type `prediction.results-00000-of-000XX`:
```json

{"instance": [26.0, 0.07885108639438881, 0.1464272109727935, 0.1828991258260679, 0.61054018345157, 0.3088025135180327, 0.19057156717160098], "prediction": 3}
{"instance": [18.0, -0.5506425558573785, -0.539030640110576, -0.542388245344349, -0.12157848914160675, -0.9829072404913923, -0.76127422027253748], "prediction": 6}
{"instance": [1.0, 0.16877874957321273, 0.20526086240043687, 0.011701852935588793, 1.3426588560447468, 0.28517943828011344, -0.489318281002783638], "prediction": 3}
{"instance": [25.0, -0.6705461067624772, -0.5936856862810885, -0.5700007087137811, -1.0977367192658425, -1.0452721591194991, -0.62529625063766058], "prediction": 1}
{"instance": [4.0, -0.4007631172260054, -0.4854167746519732, -0.41168925206237034, 0.12246106838945217, -0.4924921985521885, -0.76127422027253748], "prediction": 6}
{"instance": [8.0, -0.7005219944887517, -0.6042918471330752, -0.5810456940615539, -1.3417762767969013, -0.9507798581678222, -0.4893182810027836383], "prediction": 1}
{"instance": [11.0, -0.5506425558573785, -0.4820155843264015, -0.4172117447362567, -0.6096576042037246, -0.12397222484064802, -0.217362341733029828], "prediction": 6}
{"instance": [5.0, -0.4906907804048293, -0.5117100582300658, -0.4761183332577119, -0.3656180466726657, -0.5917091145514494, -0.62529625063766058], "prediction": 6}
{"instance": [28.0, 0.8582241672775294, 2.4080799533830155, 2.205972275359794, 0.3665006259205111, 3.2068813837059675, 2.09426314205987748], "prediction": 7}
{"instance": [2.0, -0.37078722949973075, -0.09383565010748596, -0.11347464767250347, 0.12246106838945217, 0.10186437443386016, 0.19057156717160098], "prediction": 5}
{"instance": [10.0, -0.3108354540471815, -0.4907476198969507, -0.45955085523605266, -0.3656180466726657, -0.3271306718867537, -0.62529625063766058], "prediction": 6}
{"instance": [22.0, -0.5506425558573785, -0.5144032456715388, -0.500049134844553, -0.12157848914160675, -0.6200568048369525, -0.489318281002783638], "prediction": 6}
{"instance": [13.0, -0.6705461067624772, -0.6021956032997636, -0.5681598778224857, -1.0977367192658425, -0.9980260086436606, -0.62529625063766058], "prediction": 1}
```


## FAQ

* Q: 500 Internal server error after POST request: `ValueError: [TypeError("'numpy.int64' object is not iterable"), TypeError('vars() argument must have __dict__ attribute')]`   
A: Replace last line of the `@app.post` handler. Replace `return {"predictions": [item for item in outputs]}` with `return {"predictions": outputs.tolist()}`