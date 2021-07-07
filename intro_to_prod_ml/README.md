Introduction to ML in Production

Week1 - Overview of ML Lifecycle and Development

Identify the key components of the ML Lifecycle.
Define “concept drift” as it relates to ML projects.
Differentiate between shadow, canary, and blue-green deployment scenarios in the context of varying degrees of automation.
Compare and contrast the ML modeling iterative cycle with the cycle for deployment of ML products.
List the typical metrics you might track to monitor concept drift.

Week2 - Select and Train a Model

Identify the key challenges in model development.
Describe how performance on a small set of disproportionately important examples may be more crucial than performance on the majority of examples.
Explain how rare classes in your training data can affect performance.
Define three ways of establishing a baseline for your performance.
Define structured vs. unstructured data.
Identify when to consider deployment constraints when choosing a model.
List the steps involved in getting started with ML modeling.
Describe the iterative process for error analysis.
Identify the key factors in deciding what to prioritize when working to improve model accuracy.
Describe methods you might use for data augmentation given audio data vs. image data.
Explain the problems you can have training on a highly skewed dataset.
Identify a use case in which adding more data to your training dataset could actually hurt performance.
Describe the key components of experiment tracking.

Week3 - Data definition and BaseLine

List the questions you need to answer in the process of data definition.
Compare and contrast the types of data problems you need to solve for structured vs. unstructured and big vs. small data.
Explain why label consistency is important and how you can improve it
Explain why beating human level performance is not always indicative of success of an ML model.
Make a case for improving human level performance rather than beating it.
Identify how much training data you should gather given time and resource constraints.
Describe the key steps in a data pipeline.
Compare and contrast the proof of concept vs. production phases on an ML project.
Explain the importance of keeping track of data provenance and lineage.



Introduction to Machine Learning in Production


Konstantinos, Katsiapis, Karmarkar, A., Altay, A., Zaks, A., Polyzotis, N., … Li, Z. (2020). Towards ML Engineering: A brief history of TensorFlow Extended (TFX). http://arxiv.org/abs/2010.02013 

Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2020). Challenges in deploying machine learning: A survey of case studies. http://arxiv.org/abs/2011.09926


Week 1: Overview of the ML Lifecycle and Deployment
Concept and Data Drift

Monitoring ML Models

A Chat with Andrew on MLOps: From Model-centric to Data-centric AI

Konstantinos, Katsiapis, Karmarkar, A., Altay, A., Zaks, A., Polyzotis, N., … Li, Z. (2020). Towards ML Engineering: A brief history of TensorFlow Extended (TFX). http://arxiv.org/abs/2010.02013 

Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2020). Challenges in deploying machine learning: A survey of case studies. http://arxiv.org/abs/2011.09926

Sculley, D., Holt, G., Golovin, D., Davydov, E., & Phillips, T. (n.d.). Hidden technical debt in machine learning systems. Retrieved April 28, 2021, from Nips.cc 

https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf



Week 2: Select and Train Model
Establishing a baseline

Error analysis

Experiment tracking

Brundage, M., Avin, S., Wang, J., Belfield, H., Krueger, G., Hadfield, G., … Anderljung, M. (n.d.). Toward trustworthy AI development: Mechanisms for supporting verifiable claims∗. Retrieved May 7, 2021 http://arxiv.org/abs/2004.07213v2

Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2019). Deep double descent: Where bigger models and more data hurt. Retrieved from http://arxiv.org/abs/1912.02292



Week 3: Data Definition and Baseline
Label ambiguity

https://arxiv.org/pdf/1706.06969.pdf

Data pipelines

Data lineage

MLops

Geirhos, R., Janssen, D. H. J., Schutt, H. H., Rauber, J., Bethge, M., & Wichmann, F. A. (n.d.). Comparing deep neural networks against humans: object recognition when the signal gets weaker∗. Retrieved May 7, 2021, from Arxiv.org website: