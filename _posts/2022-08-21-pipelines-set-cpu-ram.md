---
title: "Setting machine type in a Vertex AI Pipelines component"
description: "Setting machine type in a Vertex AI Custom Training job is different from Vertex AI Pipelines. You can use create_custom_training_job_from_component to set the machine type in a pipeline component "
toc: true
comments: true
layout: post
categories: ["Vertex AI"]
image: images/vertex.png
author: Rafael Sanchez
---

## Summary 

Even if both are using the same service, setting machine type, CPU and RAM in a Vertex AI Custom Training job using the Python SDK is different from setting machine type, CPU and RAM in tasks in Vertex AI Pipelines.

## Setting machine type in a Vertex AI Custom Training job 

To set machine type, CPU and RAM in a Vertex AI Custom Training job, you must use the `google.cloud.aiplatform` SDK. You can configure any type of machine according to [this table](https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types). 

To set the machine type, you must use `worker_pool_specs` and pass as argument to the method `google.cloud.aiplatform.CustomJob`:
```python
worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_K80",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_image_uri,
                "command": [],
                "args": [],
            },
        }
    ]

my_job = aiplatform.CustomJob(
    display_name='my_job',
    worker_pool_specs=worker_pool_specs,
    labels={'my_key': 'my_value'},
)

my_job.run()
```


## Setting machine type in Vertex AI Pipelines

To set machine CPU and RAM in a step in Vertex AI Training job, you must use the `kfp.v2.dsl` SDK, and the methods `set_memory_limit` and `set_cpu_limit`. Vertex AI Pipelines [will automatically find the best matching machine type to run the component](https://cloud.google.com/vertex-ai/docs/pipelines/machine-types), typically a `e2-standard` machine type.
```python
from kfp.v2 import dsl
@dsl.pipeline(name='custom-container-pipeline')
def pipeline():
  generate = generate_op()
  train = (train_op(
      training_data=generate.outputs['training_data'],
      test_data=generate.outputs['test_data'],
      config_file=generate.outputs['config_file']).
    set_cpu_limit('CPU_LIMIT').
    set_memory_limit('MEMORY_LIMIT').
    add_node_selector_constraint(SELECTOR_CONSTRAINT).
    set_gpu_limit(GPU_LIMIT))
```

However, you cannot choose a specific machine type using `set_memory_limit` and `set_cpu_limit`. Workaround is to use the [`create_custom_training_job_from_component` method](https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-1.0.4/google_cloud_pipeline_components.v1.custom_job.html#google_cloud_pipeline_components.v1.custom_job.create_custom_training_job_from_component) from `google_cloud_pipeline_components` to translate a Python component into a Vertex AI Custom Job, which allows you to specify particular Google Cloud specific machine resources, including machine types, accelerator types, among many other options.

Example:
```python
import kfp
from kfp.v2 import dsl
from kfp.v2.dsl import component
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component

# Create a normal Python component
@component(output_component_file="my_python_component.yaml", base_image="python:3.9")
def my_python_component():
  import time
  time.sleep(1)

# Convert the above component into a Custom Training job
custom_training_job = create_custom_training_job_from_component(
    my_python_component,
    display_name = 'test-component',
    machine_type = 'n1-standard-8',
    accelerator_type='NVIDIA_TESLA_P4',
    accelerator_count='1'
)

# Define a pipeline that runs the above Custom Training job
@dsl.pipeline(
  name="resource-spec-request",
  description="A simple pipeline that requests GCP machine resource",
  pipeline_root=PIPELINE_ROOT,
)
def pipeline():
  training_job_task = custom_training_job(
      project=PROJECT_ID,
      location=REGION,
  ).set_display_name('training-job-task')
```
