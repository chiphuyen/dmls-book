
# Chapter Summaries

These are the summaries of each chapter taken from the book. Some of the summaries might not make sense to readers without having first read the originating chapters, but I hope that they will give you a sense of what the book is about.

* [Chapter 1. Overview of Machine Learning Systems](#chapter-1-overview-of-machine-learning-systems)
* [Chapter 2. Introduction to Machine Learning Systems Design](#chapter-2-introduction-to-machine-learning-systems-design)
* [Chapter 3. Data Engineering Fundamentals](#chapter-3-data-engineering-fundamentals)
* [Chapter 4. Training Data](#chapter-4-training-data)
* [Chapter 5. Feature Engineering](#chapter-5-feature-engineering)
* [Chapter 6. Model Development and Offline Evaluation](#chapter-6-model-development-and-offline-evaluation)
* [Chapter 7. Model Deployment and Prediction Service](#chapter-7-model-deployment-and-prediction-service)
* [Chapter 8. Data Distribution Shifts and Monitoring](#chapter-8-data-distribution-shifts-and-monitoring)
* [Chapter 9. Continual Learning and Test in Production](#chapter-9-continual-learning-and-test-in-production)
* [Chapter 10. Infrastructure and Tooling for MLOps](#chapter-10-infrastructure-and-tooling-for-mlops)
* [Chapter 11. The Human Side of Machine Learning](#chapter-11-the-human-side-of-machine-learning)

## Chapter 1. Overview of Machine Learning Systems

This opening chapter aimed to give readers an understanding of what it takes to bring ML into the real world. We started with a tour of the wide range of use cases of ML in production today. While most people are familiar with ML in consumer-facing applications, the majority of ML use cases are for enterprise. We also discussed when ML solutions would be appropriate. Even though ML can solve many problems very well, it can’t solve all the problems and it’s certainly not appropriate for all the problems. However, for problems that ML can’t solve, it’s possible that ML can be one part of the solution.

This chapter also highlighted the differences between ML in research and ML in production. The differences include the stakeholder involvement, computational priority, the properties of data used, the gravity of fairness issues, and the requirements for interpretability. This section is the most helpful to those coming to ML production from academia. We also discussed how ML systems differ from traditional software systems, which motivated the need for this book.

ML systems are complex, consisting of many different components. Data scientists and ML engineers working with ML systems in production will likely find that focusing only on the ML algorithms part is far from enough. It’s important to know about other aspects of the system, including the data stack, deployment, monitoring, maintenance, infrastructure, etc. This book takes a system approach to developing ML systems, which means that we’ll consider all components of a system holistically instead of just looking at ML algorithms. We’ll go into detail what this holistic approach means in the next chapter.


## Chapter 2. Introduction to Machine Learning Systems Design

I hope that this chapter has given you an introduction to ML systems design and the considerations we need to take into account when designing an ML system. 

Every project must start with why this project needs to happen, and ML projects are no exception. We started the chapter with an assumption that most businesses don’t care about ML metrics unless they can move business metrics. Therefore, if an ML system is built for a business, it must be motivated by business objectives, which need to be translated into ML objectives to guide the development of ML models.

Before building an ML system, we need to understand the requirements that the system needs to meet to be considered a good system. The exact requirements vary from use case to use case, and in this chapter, we focused on the four most general requirements: reliability, scalability, maintainability, and adaptability. Techniques to satisfy each of these requirements will be covered throughout the book.

Building an ML system isn’t a one-off task but an iterative process. In this chapter, we discussed the iterative process to develop an ML system that met those above requirements.

We ended the chapter on a philosophical discussion of the role of data in ML systems. There are still many people who believe that having intelligent algorithms will eventually trump having a large amount of data. However, the success of systems including [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), [BERT](https://arxiv.org/abs/1810.04805), and [GPT](https://openai.com/blog/better-language-models/) showed that the progress of ML in the last decade relies on having access to a large amount of data. Regardless of whether data can overpower intelligent design, no one can deny the importance of data in ML. A nontrivial part of this book will be devoted to shedding light on various data questions.

Complex ML systems are made up of simpler building blocks. Now that we’ve covered the high-level overview of an ML system in production, we’ll zoom in to its building blocks in the following chapters, starting with the fundamentals of data engineering in the next chapter. If any of the challenges mentioned in this chapter seem abstract to you, I hope that specific examples in the following chapters will make them more concrete.


## Chapter 3. Data Engineering Fundamentals

This chapter is built on the foundations established in Chapter 2 around the importance of data in developing ML systems. In this chapter, we learned it’s important to choose the right format to store our data to make it easier to use the data in the future. We discussed different data formats and the pros and cons of row-major versus column-major formats as well as text versus binary formats.

We continued to cover three major data models: relational, document, and graph. Even though the relational model is the most well known given the popularity of SQL, all three models are widely used today, and each is good for a certain set of tasks.

When talking about the relational model compared to the document model, many people think of the former as structured and the latter as unstructured. The division between structured and unstructured data is quite fluid—the main question is who has to shoulder the responsibility of assuming the structure of data. Structured data means that the code that writes the data has to assume the structure. Unstructured data means that the code that reads the data has to assume the structure.

We continued the chapter with data storage engines and processing. We studied databases optimized for two distinct types of data processing: transactional processing and analytical processing. We studied data storage engines and processing together because traditionally storage is coupled with processing: transactional databases for transactional processing and analytical databases for analytical processing. However, in recent years, many vendors have worked on decoupling storage and processing. Today, we have transactional databases that can handle analytical queries and analytical databases that can handle transactional queries.

When discussing data formats, data models, data storage engines, and processing, data is assumed to be within a process. However, while working in production, you’ll likely work with multiple processes, and you’ll likely need to transfer data between them. We discussed three modes of data passing. The simplest mode is passing through databases. The most popular mode of data passing for processes is data passing through services. In this mode, a process is exposed as a service that another process can send requests for data. This mode of data passing is tightly coupled with microservice architectures, where each component of an application is set up as a service.

A mode of data passing that has become increasingly popular over the last decade is data passing through a real-time transport like Apache Kafka and RabbitMQ. This mode of data passing is somewhere between passing through databases and passing through services: it allows for asynchronous data passing with reasonably low latency.

As data in real-time transports have different properties from data in databases, they require different processing techniques, as discussed in the “Batch Processing Versus Stream Processing”** **section. Data in databases is often processed in batch jobs and produces static features, whereas data in real-time transports is often processed using stream computation engines and produces dynamic features. Some people argue that batch processing is a special case of stream processing, and stream computation engines can be used to unify both processing pipelines.


## Chapter 4. Training Data

Training data still forms the foundation of modern ML algorithms. No matter how clever your algorithms might be, if your training data is bad, your algorithms won’t be able to perform well. It’s worth it to invest time and effort to curate and create training data that will enable your algorithms to learn something meaningful.

In this chapter, we’ve discussed the multiple steps to create training data. We first covered different sampling methods, both nonprobability sampling and random sampling, that can help us sample the right data for our problem.

Most ML algorithms in use today are supervised ML algorithms, so obtaining labels is an integral part of creating training data. Many tasks, such as delivery time estimation or recommender systems, have natural labels. Natural labels are usually delayed, and the time it takes from when a prediction is served until when the feedback on it is provided is the feedback loop length. Tasks with natural labels are fairly common in the industry, which might mean that companies prefer to start on tasks that have natural labels over tasks without natural labels.

For tasks that don’t have natural labels, companies tend to rely on human annotators to annotate their data. However, hand labeling comes with many drawbacks. For example, hand labels can be expensive and slow. To combat the lack of hand labels, we discussed alternatives including weak supervision, semi-supervision, transfer learning, and active learning.

ML algorithms work well in situations when the data distribution is more balanced, and not so well when the classes are heavily imbalanced. Unfortunately, problems with class imbalance are the norm in the real world. In the following section, we discussed why class imbalance made it hard for ML algorithms to learn. We also discussed different techniques to handle class imbalance, from choosing the right metrics to resampling data to modifying the loss function to encourage the model to pay attention to certain samples.

We ended the chapter with a discussion on data augmentation techniques that can be used to improve a model’s performance and generalization for both computer vision and NLP tasks.


## Chapter 5. Feature Engineering

Because the success of today’s ML systems still depends on their features, it’s important for organizations interested in using ML in production to invest time and effort into feature engineering.

How to engineer good features is a complex question with no foolproof answers. The best way to learn is through experience: trying out different features and observing how they affect your models’ performance. It’s also possible to learn from experts. I find it extremely useful to read about how the winning teams of Kaggle competitions engineer their features to learn more about their techniques and the considerations they went through.

Feature engineering often involves subject matter expertise, and subject matter experts might not always be engineers, so it’s important to design your workflow in a way that allows nonengineers to contribute to the process.

Here is a summary of best practices for feature engineering:



* Split data by time into train/valid/test splits instead of doing it randomly.
* If you oversample your data, do it after splitting.
* Scale and normalize your data after splitting to avoid data leakage.
* Use statistics from only the train split, instead of the entire data, to scale your features and handle missing values.
* Understand how your data is generated, collected, and processed. Involve domain experts if possible.
* Keep track of your data’s lineage.
* Understand feature importance to your model.
* Use features that generalize well.
* Remove no longer useful features from your models.

With a set of good features, we’ll move to the next part of the workflow: training ML models. Before we move on, I just want to reiterate that moving to modeling doesn’t mean we’re done with handling data or feature engineering. We are never done with data and features. In most real-world ML projects, the process of collecting data and feature engineering goes on as long as your models are in production. We need to use new, incoming data to continually improve models, which we’ll cover in Chapter 9.


## Chapter 6. Model Development and Offline Evaluation

In this chapter, we’ve covered the ML algorithm part of ML systems, which many ML practitioners consider to be the most fun part of an ML project lifecycle. With the initial models, we can bring to life (in the form of predictions) all our hard work in data and feature engineering, and can finally evaluate our hypothesis (e.g., we can predict the outputs given the inputs).

We started with how to select ML models best suited for our tasks. Instead of going into pros and cons of each individual model architecture—which is a fool’s errand given the growing pools of existing models—the chapter outlined the aspects you need to consider to make an informed decision on which model is best for your objectives, constraints, and requirements.

We then continued to cover different aspects of model development. We covered not only individual models but also ensembles of models, a technique widely used in competitions and leaderboard-style research.

During the model development phase, you might experiment with many different models. Intensive tracking and versioning of your many experiments are generally agreed to be important, but many ML engineers still skip it because doing it might feel like a chore. Therefore, having tools and appropriate infrastructure to automate the tracking and versioning process is essential. We’ll cover tools and infrastructure for ML production in Chapter 10.

As models today are getting bigger and consuming more data, distributed training is becoming an essential skill for ML model developers, and we discussed techniques for parallelism including data parallelism, model parallelism, and pipeline parallelism. Making your models work on a large distributed system, like the one that runs models with hundreds of millions, if not billions, of parameters, can be challenging and require specialized system engineering expertise.

We ended the chapter with how to evaluate your models to pick the best one to deploy. Evaluation metrics don’t mean much unless you have a baseline to compare them to, and we covered different types of baselines you might want to consider for evaluation. We also covered a range of evaluation techniques necessary to sanity check your models before further evaluating your models in a production environment.

Often, no matter how good your offline evaluation of a model is, you still can’t be sure of your model’s performance in production until that model has been deployed. In the next chapter, we’ll go over how to deploy a model. 


## Chapter 7. Model Deployment and Prediction Service

Congratulations, you’ve finished possibly one of the most technical chapters in this book! The chapter is technical because deploying ML models is an engineering challenge, not an ML challenge.

We’ve discussed different ways to deploy a model, comparing online prediction with batch prediction, and ML on the edge with ML on the cloud. Each way has its own challenges. Online prediction makes your model more responsive to users’ changing preferences, but you have to worry about inference latency. Batch prediction is a workaround for when your models take too long to generate predictions, but it makes your model less flexible.

Similarly, doing inference on the cloud is easy to set up, but it becomes impractical with network latency and cloud cost. Doing inference on the edge requires having edge devices with sufficient compute power, memory, and battery.

However, I believe that most of these challenges are due to the limitations of the hardware that ML models run on. As hardware becomes more powerful and optimized for ML, I believe that ML systems will transition to making online prediction on-device.

I used to think that an ML project is done after the model is deployed, and I hope that I’ve made clear in this chapter that I was seriously mistaken. Moving the model from the development environment to the production environment creates a whole new host of problems. The first is how to keep that model in production. In the next chapter, we’ll discuss how our models might fail in production, and how to continually monitor models to detect issues and address them as fast as possible.


## Chapter 8. Data Distribution Shifts and Monitoring

This might have been the most challenging chapter for me to write in this book. The reason is that despite the importance of understanding how and why ML systems fail in production, the literature surrounding it is limited. We usually think of research preceding production, but this is an area of ML where research is still trying to catch up with production.

To understand failures of ML systems, we differentiated between two types of failures: software systems failures (failures that also happen to non-ML systems) and ML-specific failures. Even though the majority of ML failures today are non-ML-specific, as tooling and infrastructure around MLOps matures, this might change.

We discussed three major causes of ML-specific failures: production data differing from training data, edge cases, and degenerate feedback loops. The first two causes are related to data, whereas the last cause is related to system design because it happens when the system’s outputs influence the same system’s input.

We zeroed into one failure that has gathered much attention in recent years: data distribution shifts. We looked into three types of shifts: covariate shift, label shift, and concept drift. Even though studying distribution shifts is a growing subfield of ML research, the research community hasn’t yet found a standard narrative. Different papers call the same phenomena by different names. Many studies are still based on the assumption that we know in advance how the distribution will shift or have the labels for the data from both the source distribution and the target distribution. However, in reality, we don’t know what the future data will be like, and obtaining labels for new data might be costly, slow, or just infeasible.

To be able to detect shifts, we need to monitor our deployed systems. Monitoring is an important set of practices for any software engineering system in production, not just ML, and it’s an area of ML where we should learn as much as we can from the DevOps world.

Monitoring is all about metrics. We discussed different metrics we need to monitor: operational metrics—the metrics that should be monitored with any software systems such as latency, throughput, and CPU utilization—and ML-specific metrics. Monitoring can be applied to accuracy-related metrics, predictions, features, and/or raw inputs.

Monitoring is hard because even if it’s cheap to compute metrics, understanding metrics isn’t straightforward. It’s easy to build dashboards to show graphs, but it’s much more difficult to understand what a graph means, whether it shows signs of drift, and, if there’s drift, whether it’s caused by an underlying data distribution change or by errors in the pipeline. An understanding of statistics might be required to make sense of the numbers and graphs.

Detecting model performance’s degradation in production is the first step. The next step is how to adapt our systems to changing environments, which we’ll discuss in the next chapter.


## Chapter 9. Continual Learning and Test in Production

This chapter touches on a topic that I believe is among the most exciting yet underexplored topics: how to continually update your models in production to adapt them to changing data distributions. We discussed the four stages a company might go through in the process of modernizing their infrastructure for continual learning: from the manual, training from scratch stage to automated, stateless continual learning.

We then examined the question that haunts ML engineers at companies of all shapes and sizes, “How often _should_ I update my models?” by urging them to consider the value of data freshness to their models and the trade-offs between model iteration and data iteration.

Similar to online prediction discussed in Chapter 7, continual learning requires a mature streaming infrastructure. The training part of continual learning can be done in batch, but the online evaluation part requires streaming. Many engineers worry that streaming is hard and costly. It was true three years ago, but streaming technologies have matured significantly since then. More and more companies are providing solutions to make it easier for companies to move to streaming, including Spark Streaming, Snowflake Streaming, Materialize, Decodable, Vectorize, etc.

Continual learning is a problem specific to ML, but it largely requires an infrastructural solution. To be able to speed up the iteration cycle and detect failures in new model updates quickly, we need to set up our infrastructure in the right way. This requires the data science/ML team and the platform team to work together. We’ll discuss infrastructure for ML in the next chapter.


## Chapter 10. Infrastructure and Tooling for MLOps

If you’ve stayed with me until now, I hope you agree that bringing ML models to production is an infrastructural problem. To enable data scientists to develop and deploy ML models, it’s crucial to have the right tools and infrastructure set up.

In this chapter, we covered different layers of infrastructure needed for ML systems. We started from the storage and compute layer, which provides vital resources for any engineering project that requires intensive data and compute resources like ML projects. The storage and compute layer is heavily commoditized, which means that most companies pay cloud services for the exact amount of storage and compute they use instead of setting up their own data centers. However, while cloud providers make it easy for a company to get started, their cost becomes prohibitive as this company grows, and more and more large companies are looking into repatriating from the cloud to private data centers.

We then continued on to discuss the development environment where data scientists write code and interact with the production environment. Because the dev environment is where engineers spend most of their time, improvements in the dev environment translate directly into improvements in productivity. One of the first things a company can do to improve the dev environment is to standardize the dev environment for data scientists and ML engineers working on the same team. We discussed in this chapter why standardization is recommended and how to do so.

We then discussed an infrastructural topic whose relevance to data scientists has been debated heavily in the last few years: resource management. Resource management is important to data science workflows, but the question is whether data scientists should be expected to handle it. In this section, we traced the evolution of resource management tools from cron to schedulers to orchestrators. We also discussed why ML workflows are different from other software engineering workflows and why they need their own workflow management tools. We compared various workflow management tools such as Airflow, Argo, and Metaflow.

ML platform is a team that has emerged recently as ML adoption matures. Since it’s an emerging concept, there are still disagreements on what an ML platform should consist of. We chose to focus on the three sets of tools that are essential for most ML platforms: deployment, model store, and feature store. We skipped monitoring of the ML platform since it’s already covered in Chapter 8.

When working on infrastructure, a question constantly haunts engineering managers and CTOs alike: build or buy? We ended this chapter with a few discussion points that I hope can provide you or your team with sufficient context to make those difficult decisions.


## Chapter 11. The Human Side of Machine Learning

Despite the technical nature of ML solutions, designing ML systems can’t be confined in the technical domain. They are developed by humans, used by humans, and leave their marks in society. In this chapter, we deviated from the technical theme of the last eight chapters to focus on the human side of ML.

We first focused on how the probabilistic, mostly correct, and high-latency nature of ML systems can affect user experience in various ways. The probabilistic nature can lead to inconsistency in user experience, which can cause frustration—“Hey, I just saw this option right here, and now I can’t find it anywhere.” The mostly correct nature of an ML system might render it useless if users can’t easily fix these predictions to be correct. To counter this, you might want to show users multiple “most correct” predictions for the same input, in the hope that at least one of them will be correct.

Building an ML system often requires multiple skill sets, and an organization might wonder how to distribute these required skill sets: to involve different teams with different skill sets or to expect the same team (e.g., data scientists) to have all the skills. We explored the pros and cons of both approaches. The main cons of the first approach is overhead in communication. The main cons of the second approach is that it’s difficult to hire data scientists who can own the process of developing an ML system end-to-end. Even if they can, they might not be happy doing it. However, the second approach might be possible if these end-to-end data scientists are provided with sufficient tools and infrastructure, which was the focus of Chapter 10.

We ended the chapter with what I believe to be the most important topic of this book: responsible AI. Responsible AI is no longer just an abstraction, but an essential practice in today’s ML industry that merits urgent actions. Incorporating ethics principles into your modeling and organizational practices will not only help you distinguish yourself as a professional and cutting-edge data scientist and ML engineer but also help your organization gain trust from your customers and users. It will also help your organization obtain a competitive edge in the market as more and more customers and users emphasize their need for responsible AI products and services.

It is important to not treat this responsible AI as merely a checkbox ticking activity that we undertake to meet compliance requirements for our organization. It’s true that the framework proposed in this chapter will help you meet the compliance requirements for your organization, but it won’t be a replacement for critical thinking on whether a product or service should be built in the first place.
