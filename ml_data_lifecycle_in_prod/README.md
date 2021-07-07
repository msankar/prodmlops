# Machine Learning Data Lifeycle in Production


## Week1 - Collecting, Labeling and Validating Data
* Describe the differences between ML modeling and a production ML system
* Identify responsible data collection for building a fair production ML system
* Discuss data and concept change and how to address it by annotating new training data with direct labeling and/or human labeling
* Address training data issues by generating dataset statistics and creating, comparing and updating data schemasa

## Week2 - Feature Engineering, Transformation and Selection
* Define a set of feature engineering techniques, such as scaling and binning
* Use TensorFlow Transform for a simple preprocessing and data transformation task
* Describe feature space coverage and implement different feature selection methods
* Perform feature selection using scikit-learn routines and ensure feature space coverage

## Week3 - Data Journey and Data Storage
* Describe data journey through data lineage and provenance
* Integrate the sequence of pipeline artifacts into metadata storage using ML Metadata library
* Iteratively create enterprise data schema
* Explain how to integrate enterprise data into feature stores, data warehouses and data lakes

## Week4 - Advanced Labeling, Augmentation and Data Preprocessing
* Discuss direct, semi-supervised, weak supervision and active learning methods for labeling data
* Increase the diversity of your training set by data augmentation
* Perform advanced data preparation and transformation on different structured and unstructured data types

## Reading
Konstantinos, Katsiapis, Karmarkar, A., Altay, A., Zaks, A., Polyzotis, N., … Li, Z. (2020). Towards ML Engineering: A brief history of TensorFlow Extended (TFX). http://arxiv.org/abs/2010.02013 

Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2020). Challenges in deploying machine learning: A survey of case studies. http://arxiv.org/abs/2011.09926

#### Week 1: Collecting, Labeling and Validating Data 
ML code fraction:

[MLops](https://cd.foundation/blog/2020/02/11/announcing-the-cd-foundation-mlops-sig/)

[Data 1st class citizen](https://karpathy.medium.com/software-2-0-a64152b37c35)

[Runners app](https://pair.withgoogle.com/chapter/data-collection/)

[Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)

[Bias in datasets](https://ai.googleblog.com/2018/09/introducing-inclusive-images-competition.html)

[Logstash](https://www.elastic.co/logstash/)

[Fluentd](https://www.fluentd.org/)

[Google Cloud Logging](https://cloud.google.com/logging/)

AWS ElasticSearch https://aws.amazon.com/elasticsearch-service/

Azure Monitor https://azure.microsoft.com/en-us/services/monitor/

TFDV https://blog.tensorflow.org/2018/09/introducing-tensorflow-data-validation.html

[Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance)

Sculley, D., Holt, G., Golovin, D., Davydov, E., & Phillips, T. (n.d.). Hidden technical debt in machine learning systems. Retrieved April 28, 2021, from Nips.cc https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf


#### Week 2: Feature Engineering, Transformation and Selection
Mapping raw data into feature - https://developers.google.com/machine-learning/crash-course/representation/feature-engineering

Feature engineering techniques- https://www.commonlounge.com/discussion/3ce75d036e924c70ab7e47f534ec40fc/history

Facets -  https://pair-code.github.io/facets/

Embedding projector - http://projector.tensorflow.org/

Encoding features - https://developers.google.com/machine-learning/crash-course/feature-crosses/encoding-nonlinearity

TFX:

1. https://www.tensorflow.org/tfx/guide#tfx_pipelines
2. https://ai.googleblog.com/2017/02/preprocessing-for-machine-learning-with.html
Breast Cancer Dataset - http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29

#### Week 3: Data Journey and Data Storage 
##### Data Versioning:

https://dvc.org/
https://git-lfs.github.com/

##### ML Metadata:

https://www.tensorflow.org/tfx/guide/mlmd#data_model
https://www.tensorflow.org/tfx/guide/understanding_custom_components

##### Chicago taxi trips data set: 

https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew/data
https://archive.ics.uci.edu/ml/datasets/covertype

##### Feast:

https://cloud.google.com/blog/products/ai-machine-learning/introducing-feast-an-open-source-feature-store-for-machine-learning
https://github.com/feast-dev/feast
https://blog.gojekengineering.com/feast-bridging-ml-models-and-data-efd06b7d1644

#### Week 4: Advanced Labeling, Augmentation and Data Preprocessing
[Hand Labeling](https://twitter.com/jeffdean/status/1106325994913189888?lang=en)

[Weak supervision](https://dawn.cs.stanford.edu/2017/07/16/weak-supervision/)

[Snorkel](https://www.snorkel.org/)

[How do you get more data?](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html)

[Advanced Techniques](https://github.com/google-research/uda)

[Images in tensorflow](https://www.tensorflow.org/lite/examples/image_classification/overview)

CIFAR-10

https://www.cs.toronto.edu/~kriz/cifar.html
https://www.tensorflow.org/datasets/catalog/cifar10

[Weather dataset](https://www.bgc-jena.mpg.de/wetter/)

Human Activity Recognition - https://www.cis.fordham.edu/wisdm/dataset.php

Papers

Label Propagation:

Iscen, A., Tolias, G., Avrithis, Y., & Chum, O. (2019). Label propagation for deep semi-supervised learning. https://arxiv.org/pdf/1904.04717.pdf
