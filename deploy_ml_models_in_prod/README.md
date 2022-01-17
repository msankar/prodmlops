# Deploying Machine Learning Models in Production

## 1 - Model Serving: Introduction
* Identify and contrast the challenges for serving inference requests
* Compare cost, latency and throughput metrics to optimize serving inference requests
* Judge the hardware resources and requirements for your serving models so that your system is reliable and can scale based on demand
* Install and Use TensorFlow Serving to serve inference requests on a simple image classification model

## 2 - Model Serving: Patterns and Infrastructure
* Serve models and deliver inference results by building scalable and reliable infrastructure.
* Contrast the use case for batch and realtime inference and how to optimize performance and hardware usage in each case
* Implement techniques to run inference on both edge devices and applications running in a web browser
* Outline and structure your data preprocessing pipeline to match your inference requirements
* Distinguish the performance and resource requirements for static and stream based batch inference

## 3 - Model Management and Delivery
* Coordinate model management, tracking and delivery by managing model versions, lineage, and registries
* Implement ML processes, pipelines, and workflow automation that adhere to modern MLOps practices
* Carry out continuous integration and delivery for model deployment, and maintain and monitor a continuously operating production system

## 4 - Model Monitoring and Logging
* Manage, monitor and audit your projects during their entire lifecycle
* Establish procedures to detect model decay and prevent reduced accuracy
* Implement modern practices to trace your ML system including monitoring and logging
* Integrate mechanisms to your production systems to ensure that they comply with modern responsible AI practices and legal regulations
* Determine when is appropriate and how to apply techniques to protect users privacy and respect their right to be forgotten


## READING
## Week 1. Model Serving: introduction
### NoSQL Databases:
[Google Cloud Memorystore](https://cloud.google.com/memorystore)

[Google Cloud Firestore](https://cloud.google.com/firestore)

[Google Cloud Bigtable](https://cloud.google.com/bigtable)

[Amazon DynamoDB](https://aws.amazon.com/dynamodb/)

### MobileNets:
[MobileNets](https://arxiv.org/abs/1704.04861)

### Serving Systems:
[Clipper](https://rise.cs.berkeley.edu/projects/clipper/)

[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)

##  Week 2. Model Serving: patterns and infrastructure
### Model Serving Architecture:
[Model Server Architecture](https://medium.com/@vikati/the-rise-of-the-model-servers-9395522b6c58)

[TensorFlow Serving](https://www.tensorflow.org/tfx/serving/architecture)

[NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)

[Torch Serve](https://github.com/pytorch/serve)

[Kubeflow Serving](https://www.kubeflow.org/docs/components/serving/)


### Scaling Infrastructure:
[Container Orchestration](https://phoenixnap.com/blog/what-is-container-orchestration)

[Kubernetes](https://kubernetes.io/)

[Docker Swarm](https://docs.docker.com/engine/swarm/)

[Kubeflow](https://www.kubeflow.org/)

### Online Inference:
[Batch vs. Online Inference](https://mlinproduction.com/batch-inference-vs-online-inference/)

### Batch Processing with ETL:
[Kafka ML](https://github.com/ertis-research/kafka-ml#:~:text=Kafka%2DML%20is%20a%20framework,(ML)%20models%20on%20Kubernetes.&text=The%20training%20and%20inference%20datasets,ones%20provided%20by%20the%20IoT.)

[Pub Sub](https://cloud.google.com/pubsub)

[Cloud DataFlow](https://cloud.google.com/dataflow)

[Apache Spark](https://spark.apache.org/)

## Week 3. Model Management and Delivery
### Experiment Tracking and Management:
[Tracking](https://towardsdatascience.com/machine-learning-experiment-tracking-93b796e501b0)

[Management](https://neptune.ai/blog/experiment-management)

### Notebooks:
[nbconvert](https://nbconvert.readthedocs.io/)

[nbdime](https://nbdime.readthedocs.io/)

[jupytext](https://jupytext.readthedocs.io/en/latest/install.html)

[neptune-notebooks](https://docs.neptune.ai/)

[git](https://git-scm.com/)

### Tools for Data Versioning:
[Neptune](https://docs.neptune.ai/how-to-guides/data-versioning)

[Pachyderm](https://www.pachyderm.com/)

[Delta Lake](https://delta.io/)

[Git LFS](https://git-lfs.github.com/)

[DoIt](https://github.com/dolthub/dolt)

[lakeFS](https://lakefs.io/data-versioning/)

[DVC](https://dvc.org/)

[ML-Metadata](https://blog.tensorflow.org/2021/01/ml-metadata-version-control-for-ml.html)

### Tooling for Teams:
[Image Summaries](https://www.tensorflow.org/tensorboard/image_summaries)

[neptune-ai](https://neptune.ai/for-teams)

[Vertex TensorBoard](https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-overview)

### MLOps:
[MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Orchestrated Workflows with TFX:
[Creating a Custom TFX Component](https://blog.tensorflow.org/2020/01/creating-custom-tfx-component.html)

[Building Fully Custom Components](https://github.com/tensorflow/tfx/blob/master/docs/guide/custom_component.md)

### Continuous and Progressive Delivery:
[Progressive Delivery](https://www.split.io/glossary/progressive-delivery/)

[Continuous, Incremental, & Progressive Delivery](https://launchdarkly.com/blog/continuous-incrementalprogressive-delivery-pick-three/)

[Deployment Strategies](https://dev.to/mostlyjason/intro-to-deployment-strategies-blue-green-canary-and-more-3a3)

[Blue/Green Deployment](https://martinfowler.com/bliki/BlueGreenDeployment.html)

[A/B Testing](https://medium.com/capital-one-tech/the-role-of-a-b-testing-in-the-machine-learning-future-3d2ba035daeb)

## Week 4. Model Monitoring and Logging
[Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

[Monitoring Machine Learning Models in Production](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)

[Google Cloud Monitoring](https://cloud.google.com/monitoring)

[Amazon CloudWatch](https://aws.amazon.com/cloudwatch/)

[Azure Monitor](https://docs.microsoft.com/en-us/azure/azure-monitor/overview#:~:text=Azure%20Monitor%20helps%20you%20maximize,cloud%20and%20on%2Dpremises%20environments.&text=Collect%20data%20from%20monitored%20resources%20using%20Azure%20Monitor%20Metrics.)

[Dapper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36356.pdf)

[Jaeger](https://www.jaegertracing.io/)

[Zipkin](https://zipkin.io/)

[Vertex Prediction](https://cloud.google.com/vertex-ai)

[Vertex Labelling Service](https://cloud.google.com/vertex-ai/docs/datasets/label-using-console)

[How “Anonymous” is Anonymized Data?](https://www.kdnuggets.com/2020/08/anonymous-anonymized-data.html)

[Pseudonymization](https://dataprivacymanager.net/pseudonymization-according-to-the-gdpr/)


