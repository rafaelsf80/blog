---
title: "On Token Generation inside Vertex AI Training and Pipelines"
description: "Vertex AI single replica training doesn't support token generation on BYOSA usage. This post shows a workaorund using iamcredentials.googleapis.com"
toc: true
comments: true
layout: post
categories: ["Vertex AI"]
image: images/vertex.png
author: Rafael Sanchez
---

## Summary

As stated [here](https://issuetracker.google.com/issues/229537245), Vertex AI Training doesn't support token generation when using BYOSA. This affects standalone training jobs as well as Vertex AI Pipelines tasks. This includes also calling `fetch_id_token()` in a Vertex AI Training job.  **Root cause** is that Vertex AI training **can not reach the compute metadata service required to retrieve a token**. The workaround is to use `iamcredentials.googleapis.com`. 
This only happens in Vertex AI single replica training, not in Kubernetes training.


## Context

There are two types of tokens (authentication and authorization):
* **Identity token:** (authentication) the identity token is used when calling other Cloud Run services or when invoking any service that can validate an identity token. It used to be fetched from the VM metadata service, and it is included in the `Authorization: Bearer` header. **IAP requires Identity tokens** (Cloud IAM) supported by App Engine, GCE, GKE and Cloud Run. For example, identity tokens are retrieved for App Engine using `fetch_id_token`.
* **Access token:** (authorization) the access token is used for OAuth 2.0 when calling most Google APIs.
It used to be fetched from the VM metadata service as well. 

Google's OAuth 2.0 APIs can be used [for both authentication and authorization](https://developers.google.com/identity/protocols/oauth2/openid-connect) using the **OpenID Connect (OIDC) standard**.

To achieve BYOSA (allow running workloads with user’s service account) on GCE training jobs, there are two metadata servers on a GCE VM:
1. Cloudbuild metadata server: it contains the access token for the user service account.
2. GCE metadata server.

All the customer’s requests (sending to the GCE metadata server) are redirected to the CloudBuild metadata server for the BYOSA usage, however, that metadata server doesn’t support fetching the **identity token**. The customer gets a 401 error code.


## Calling Cloud Run or Cloud Functions from within Vertex AI Pipelines

Both Cloud Run and Cloud Functions are deployed in private mode by default, protected by Cloud IAM. This means an unauthenticated user **can not invoke** the service: only an authenticated user with the role `roles/run.invoker` or `roles/cloudfunctions.invoker`, respectively, can do it.

To retrieve a token with Cloud IAM inside Vertex AI Pipelines, and overcome the limitation to get a token in Vertex AI Training job, you can use `iamcredentials.googleapis.com` inside Vertex AI Training.

In the code example, there are two separate service accounts (SAs) in the Vertex AI pipeline (recommended to be different, but could be the same)

* SA #1 (pipeline): SA for the pipeline run.
* SA #2 (component): SA for service account requesting permissions to access a separate services like Cloud Run, App Engine, ...

First, you use `google.auth.default()` to get your credentials. Then, there are two HTTP calls:
1. **HTTP call #1:** those credentials generate an access token used to call `iamcredentials.googleapis.com` through the class `google.auth.transport.requests.AuthorizedSession` (if you sniff the HTTP traffic, that access token is on the `Authorization: bearer` header). You must set the `audience` field to the service URL. You need the `iam.serviceAccounts.getAccessToken` IAM permission on the service account #2. This permission can be granted in the GCP IAM console (for example: the role `Cloud Run Service Agent` or the role `Service Account Token Creator` contains that IAM permission) or using `gcloud iam service-accounts add-iam-policy-binding` with the option `--role=roles/iam.serviceAccountTokenCreator`.

2. **HTTP call #2:** the previous response contains a new OIDC Token, which is used in the `Authorization: bearer` to call Cloud Run or Cloud Function service.

To summarize: you are using one Access Token to create another Access Token (via the `iamcredentials.googleapis.com` API call) changing identities. But you need the role `roles/iam.serviceAccountTokenCreator` in the  service account to be able to obtain the second Access Token. Additionally, you must add the proper permissions to call other services, like Cloud Run or Cloud Functions (for example: `Cloud Run Service Agent` to call Cloud Run).

Permissions required for the service account:
* `roles/iam.serviceAccountTokenCreator`
* `roles/run.invoker` or `roles/cloudfunctions.invoker`, included in **Cloud Run Service Agent** or **Cloud Function Service Agent** role.

```py
import google.cloud.aiplatform as aip
from kfp.v2 import compiler
from kfp.v2 import dsl
from kfp.v2.dsl import (component)

project_id = 'MY_PROJECT_ID'  #<---- **CHANGE ME**
pipeline_root_path = 'gs://BUCKET/FOLDER' #<---- **CHANGE ME**
template_name = 'my_template.json'  #<---- **CHANGE ME**
service_account_pipeline = 'debug-token-fetch@MY_PROJECT_ID.iam.gserviceaccount.com'  #<---- **CHANGE ME**
region = 'MY_REGION'  #<---- **CHANGE ME**

@component
def authed_task():
    import google.auth
    import json
    from google.auth.transport.requests import AuthorizedSession
    import urllib
    
    service_account_req = 'debug-token-fetch@MY_PROJECT_ID.iam.gserviceaccount.com'  #<---- **CHANGE ME**
    service_url = 'https://my_cloud_run_service'   #<---- **CHANGE ME**

    # Gets the access token for service_account.
    credentials, projectid = google.auth.default()

    # do credentials.refresh() to guarantee service_account_email is correctly updated
    # as per https://google-auth.readthedocs.io/en/master/reference/google.auth.compute_engine.credentials.html
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)

    if hasattr(credentials, "service_account_email"):
            service_account_email_name = credentials.service_account_email
            print(f'Service account: {service_account_email_name}')
    
    audience = service_url

    iamcredentials_url = f'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{service_account_req}:generateIdToken'
        
    token_headers = {'content-type': 'application/json'}
    body = json.dumps({'audience': audience, 'includeEmail': 'TRUE'})

    authed_session = AuthorizedSession(credentials)
    
    # Requests the identity token from 'iamcredentials.googleapis.com' through the access token.
    token_response = authed_session.request('POST', iamcredentials_url, data=body, headers=token_headers).json()
    print(token_response)
    oidc_token = token_response['token']

    data = '**Hello Bold Text**'

    req = urllib.request.Request(service_url, data=data.encode())

    req.add_header("Authorization", f"Bearer {oidc_token}")
    response = urllib.request.urlopen(req)

    print('{}'.format(response.read()))

              
@dsl.pipeline(
    name="auth-pipeline-cloudrun",
    description="pipeline to debug token fetch.",
    pipeline_root=pipeline_root_path
)
def debug_auth_pipeline():
    debug_out = authed_task()

if __name__ == "__main__":
    
    compiler.Compiler().compile(
        pipeline_func=debug_auth_pipeline,
        package_path=template_name
    )
    
    aip.init(
    project=project_id,
    location=region,
    )

    job = aip.PipelineJob(
        display_name='AuthDebugJobCloudRun',
        template_path=template_name,
        pipeline_root=pipeline_root_path,
        enable_caching=False,
    )
   
    job.run(service_account=service_account_pipeline)
```


## Calling App Engine (IAP) from within Vertex AI Pipelines

To retrieve a token for a service behind IAP, like App Engine, you need to fetch the token using `fetch_id_token`, as described [here](https://cloud.google.com/iap/docs/authentication-howto#authenticating_from_a_service_account). However, this fails with a 401 error for the same reason as above due to the limitation to get a token in a Vertex AI Training job.

To overcome the limitation again, you  use `iamcredentials.googleapis.com` inside Vertex AI Training, putting the `cliend_id` in the `audience` field.

Permissions required for the service account:
* `roles/iam.serviceAccountTokenCreator`
* `roles/iap.httpsResourceAccessor`, included in **IAP-secured Web App User** role.

```py
import google.cloud.aiplatform as aip
from kfp.v2 import compiler
from kfp.v2 import dsl
from kfp.v2.dsl import (component)

project_id = 'MY_PROJECT_ID'  #<---- **CHANGE ME**
pipeline_root_path = 'gs://BUCKET/FOLDER' #<---- **CHANGE ME**
template_name = 'my_template.json'  #<---- **CHANGE ME**
service_account_pipeline = 'debug-token-fetch@MY_PROJECT_ID.iam.gserviceaccount.com'  #<---- **CHANGE ME**
region = 'MY_REGION'  #<---- **CHANGE ME** 

@component
def authed_task():
    import google.auth
    import json
    from google.auth.transport.requests import AuthorizedSession
    import urllib, requests
    
    service_account_req = 'debug-token-fetch@MY_PROJECT_ID.iam.gserviceaccount.com'   #<---- **CHANGE ME**
    service_url = 'https://my_app_engine_service_nehing_iap'     #<---- **CHANGE ME**

    # Gets the access token for service_account.
    credentials, projectid = google.auth.default()

    # do credentials.refresh() to guarantee service_account_email is correctly updated
    # as per https://google-auth.readthedocs.io/en/master/reference/google.auth.compute_engine.credentials.html
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)

    if hasattr(credentials, "service_account_email"):
            service_account_email_name = credentials.service_account_email
            print(f'Service account: {service_account_email_name}')

    audience = 'CLIEND_ID'    #<---- **CHANGE ME**

    iamcredentials_url = f'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{service_account_req}:generateIdToken'
        
    token_headers = {'content-type': 'application/json'}
    body = json.dumps({'audience': audience, 'includeEmail': 'TRUE'})

    authed_session = AuthorizedSession(credentials)
    
    # Requests the identity token from 'iamcredentials.googleapis.com' through the access token.
    token_response = authed_session.request('POST', iamcredentials_url, data=body, headers=token_headers).json()
    oidc_token = token_response['token']

    data = '**Hello Bold Text**'

    req = urllib.request.Request(service_url, data=data.encode())

    req.add_header("Authorization", f"Bearer {oidc_token}")
    response = urllib.request.urlopen(req)

    print('{}'.format(response.read()))
 
             
@dsl.pipeline(
    name="auth-pipeline-iap",
    description="pipeline to debug token fetch.",
    pipeline_root=pipeline_root_path
)
def debug_auth_pipeline():
    debug_out = authed_task()

if __name__ == "__main__":
    
    compiler.Compiler().compile(
        pipeline_func=debug_auth_pipeline,
        package_path=template_name
    )
    
    aip.init(
    project=project_id,
    location=region,
    )

    job = aip.PipelineJob(
        display_name='AuthDebugJobCloudRun',
        template_path=template_name,
        pipeline_root=pipeline_root_path,
        enable_caching=False,
    )
   
    job.run(service_account=service_account_pipeline) 
```

## Notes

* As per [public SDK docs of `google.auth`](https://google-auth.readthedocs.io/en/master/reference/google.auth.html), the method `google.auth.default()` gets credentials differently is you are launching from local or from within a cloud service (App Engine, GCE, Vertex, ...). For execution in Vertex AI, credentials are obtained from the metadata service (see section 4 in the docs). However, if local, credentials are obtained from either `GOOGLE_APPLICATION_CREDENTIALS` or **gcloud SDK** (see sections 1-2). 
* As per [public SDK docs of `google.auth.compute_engine.credentials.Credentials`]( https://google-auth.readthedocs.io/en/master/reference/google.auth.compute_engine.credentials.html), it is stated that the `service_account_email` field is not guaranteed until a `refresh()` is done. In case you get the string `default` instead of a valid email service account, make sure you call `refresh()`. 
```py
credentials, projectid = google.auth.default()

request = google.auth.transport.requests.Request()
credentials.refresh(request)

if hasattr(credentials, "service_account_email"):
   service_account_email_name = credentials.service_account_email
   print(f'Service account: {service_account_email_name}')
```
* In case you launch the code locally, not in Vertex AI, you must: 1) Use a JSON key and point it to GOOGLE_APPLICATION_CREDENTIALS; 2) Make sure you add auth scopes when calling `google.auth.default`: `credentials, projectid = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])`


## References

`[1]` [Authentication from a service acoount to an IAP-web service](https://cloud.google.com/iap/docs/authentication-howto#authenticating_from_a_service_account)    
`[2]` [Medium article](https://medium.com/google-cloud/secure-cloud-run-cloud-functions-and-app-engine-with-api-key-73c57bededd1) on Securing Cloud Run, Cloud Functions and App Engine with an API key     
`[3]` https://stackoverflow.com/questions/67640123/how-to-protect-app-hosted-on-google-cloud-run   