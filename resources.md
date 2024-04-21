# Resources

The resources here are meant for further exploration of topics already covered in the book. Some of them were excluded from the book to avoid distracting the readers from the key points, as the book already includes a substantial amount of links and references.

* [Chapter 1. Overview of Machine Learning Systems](#chapter-1-overview-of-machine-learning-systems)
* [Chapter 2. Introduction to Machine Learning Systems Design](#chapter-2-introduction-to-machine-learning-systems-design)
* [Chapter 3. Data Engineering Fundamentals](#chapter-3-data-engineering-fundamentals)
    * [Streaming systems](#streaming-systems)
* [Chapter 4. Training Data](#chapter-4-training-data)
* [Chapter 5. Feature Engineering](#chapter-5-feature-engineering)
* [Chapter 6. Model Development and Offline Evaluation](#chapter-6-model-development-and-offline-evaluation)
    * [Training, debugging, and testing ML code](#training-debugging-and-testing-ml-code)
    * [Model evaluation](#model-evaluation)
* [Chapter 7. Model Deployment and Prediction Service](#chapter-7-model-deployment-and-prediction-service)
* [Chapter 8. Data Distribution Shifts and Monitoring](#chapter-8-data-distribution-shifts-and-monitoring)
* [Chapter 9. Continual Learning and Test in Production](#chapter-9-continual-learning-and-test-in-production)
    * [Contextual bandits](#contextual-bandits)
* [Chapter 10. Infrastructure and Tooling for MLOps](#chapter-10-infrastructure-and-tooling-for-mlops)
* [Chapter 11. The Human Side of Machine Learning](#chapter-11-the-human-side-of-machine-learning)

## Chapter 1. Overview of Machine Learning Systems

To learn to design ML systems, it’s helpful to read case studies to see how actual teams deal with different deployment requirements and constraints. Many companies — Airbnb, Lyft, Uber, and Netflix, to name a few — run excellent tech blogs where they share their experience using ML to improve their products and/or processes.

1. [Using Machine Learning to Predict Value of Homes On Airbnb](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d) (Robert Chang, Airbnb Engineering & Data Science, 2017)

In this detailed and well-written blog post, Chang described how Airbnb used machine learning to predict an important business metric: the value of homes on Airbnb. It walks you through the entire workflow: feature engineering, model selection, prototyping, moving prototypes to production. It's completed with lessons learned, tools used, and code snippets too.

2. [Using Machine Learning to Improve Streaming Quality at Netflix](https://medium.com/netflix-techblog/using-machine-learning-to-improve-streaming-quality-at-netflix-9651263ef09f) (Chaitanya Ekanadham, Netflix Technology Blog, 2018)

As of 2018, Netflix streams to over 117M members worldwide, half of those living outside the US. This blog post describes some of their technical challenges and how they use machine learning to overcome these challenges, including to predict the network quality, detect device anomaly, and allocate resources for predictive caching.

3. [150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com](https://blog.kevinhu.me/2021/04/25/25-Paper-Reading-Booking.com-Experiences/bernardi2019.pdf) (Bernardi et al., KDD, 2019)

As of 2019, Booking.com has around 150 machine learning models in production. These models solve a wide range of prediction problems (e.g. predicting users’ travel preferences and how many people they travel with) and optimization problems (e.g.optimizing the background images and reviews to show for each user). Adrian Colyer gave a good summary of the six lessons learned [here](https://blog.acolyer.org/2019/10/07/150-successful-machine-learning-models/):


1. Machine learned models deliver strong business value.
2. Model performance is not the same as business performance.
3. Be clear about the problem you’re trying to solve.
4. Prediction serving latency matters.
5. Get early feedback on model quality.
6. Test the business impact of your models using randomized controlled trials.

4. [How we grew from 0 to 4 million women on our fashion app, with a vertical machine learning approach](https://medium.com/hackernoon/how-we-grew-from-0-to-4-million-women-on-our-fashion-app-with-a-vertical-machine-learning-approach-f8b7fc0a89d7) (Gabriel Aldamiz, HackerNoon, 2018)

To offer automated outfit advice, Chicisimo tried to qualify people's fashion taste using machine learning. Due to the ambiguous nature of the task, the biggest challenges are framing the problem and collecting the data for it, both challenges are addressed by the article. It also covers the problem that every consumer app struggles with: user retention.

5. [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) (Mihajlo Grbovic, Airbnb Engineering & Data Science, 2019)

This article walks you step by step through a canonical example of the ranking and recommendation problem. The four main steps are system design, personalization, online scoring, and business aspects. The article explains which features to use, how to collect data and label it, why they chose Gradient Boosted Decision Tree, which testing metrics to use, what heuristics to take into account while ranking results, and how to do A/B testing during deployment. Another wonderful thing about this post is that it also covers personalization to rank results differently for different users. 

6. [From shallow to deep learning in fraud](https://eng.lyft.com/from-shallow-to-deep-learning-in-fraud-9dafcbcef743) (Hao Yi Ong, Lyft Engineering, 2018)

Fraud detection is one of the earliest use cases of machine learning in the industry. This article explores the evolution of fraud detection algorithms used at Lyft. At first, an algorithm as simple as logistic regression with engineered features was enough to catch most fraud cases. Its simplicity allowed the team to understand the importance of different features. Later, when fraud techniques have become too sophisticated, more complex models are required. This article explores the tradeoff between complexity and interpretability, performance and ease of deployment.

7. [Space, Time and Groceries](https://tech.instacart.com/space-time-and-groceries-a315925acf3a) (Jeremy Stanley, Tech at Instacart, 2017)

Instacart uses machine learning to solve the task of path optimization: how to most efficiently assign tasks for multiple shoppers and find the optimal paths for them.  The article explains the entire process of system design, from framing the problem, collecting data, algorithm and metric selection, topped with a tutorial for beautiful visualization.

8. [Creating a Modern OCR Pipeline Using Computer Vision and Deep Learning](https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/) (Brad Neuberg, Dropbox Engineering, 2017)

An application as simple as a document scanner has two distinct components: optical character recognition and word detector. Each requires its own production pipeline, and the end-to-end system requires additional steps for training and tuning. This article also goes into detail the team’s effort to collect data, which includes building their own data annotation platform.

9. [Spotify’s Discover Weekly: How machine learning finds your new music](https://hackernoon.com/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe) (Sophia Ciocca, 2017)

To create Discover Weekly, there are three main types of recommendation models that Spotify employs:



* **Collaborative Filtering **models (i.e. the ones that Last.fm originally used), which work by analyzing your behavior and others’ behavior.
* **Natural Language Processing** (NLP) models, which work by analyzing text.
* **Audio** models, which work by analyzing the raw audio tracks themselves.

10. [Smart Compose: Using Neural Networks to Help Write Emails](https://ai.googleblog.com/2018/05/smart-compose-using-neural-networks-to.html) (Yonghui Wu, Google AI Blog 2018)

“_Since Smart Compose provides predictions on a per-keystroke basis, it must respond ideally within **100ms** for the user not to notice any delays. Balancing model complexity and inference speed was a critical issue_.”


## Chapter 2. Introduction to Machine Learning Systems Design

* [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) (Martin Zinkevich)
* [Things I wish we had known before we started our first Machine Learning project](https://medium.com/infinity-aka-aseem/things-we-wish-we-had-known-before-we-started-our-first-machine-learning-project-336d1d6f2184) (Aseem Bansal, towards-infinity 2018)
* [Data Science Project Quick-Start](https://eugeneyan.com/writing/project-quick-start/) (Eugene Yan, 2022)
* [https://github.com/chiphuyen/machine-learning-systems-design](https://github.com/chiphuyen/machine-learning-systems-design): A much earlier, much less organized version of this book. 
* [Deploying Machine Learning Models: A Checklist](https://twolodzko.github.io/ml-checklist) (a short checklist for ML systems design)


## Chapter 3. Data Engineering Fundamentals

* [A Beginner’s Guide to Data Engineering](https://medium.com/@rchang/a-beginners-guide-to-data-engineering-part-i-4227c5c457d7) (Robert Chang 2018)
* [Designing Data-Intensive Applications](https://learning.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) (Martin Kleppmann, O’Reilly, 2017)
* [Emerging Architectures for Modern Data Infrastructure](https://future.a16z.com/emerging-architectures-modern-data-infrastructure/) (Bornstein et al, a16z 2022) 
* [Reverse ETL — A Primer](https://medium.com/memory-leak/reverse-etl-a-primer-4e6694dcc7fb) (Astasia Myers 2021)
* [Uber’s Big Data Platform: 100+ Petabytes with Minute Latency](https://eng.uber.com/uber-big-data-platform/) (Reza Shiftehfar, Uber Engineering blog 2018)
* [How DoorDash is Scaling its Data Platform to Delight Customers and Meet our Growing Demand](https://doordash.engineering/2020/09/25/how-doordash-is-scaling-its-data-platform/) (Sudhir Tonse 2020)


### Streaming systems

* [The Log: What every software engineer should know about real-time data's unifying abstraction](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying) (Jay Kreps, LinkedIn / Confluent, 2013): Jay mentioned in a [tweet](https://twitter.com/jaykreps/status/1408159236794765314) that when he wrote the blog to see if there was enough interest in streaming for his team to start a company around it. The blog must have been popular because his team spun out of LinkedIn to become Confluent.
* [The Many Meanings of Event-Driven Architecture](https://www.youtube.com/watch?v=STKCRSUsyP0) (Martin Fowler, GOTO 2017): Martin Fowler is a great speaker. His talk made clear many of the complexities of event-driven architecture.
* [Stream Processing Hard Problems – Part 1: Killing Lambda](https://engineering.linkedin.com/blog/2016/06/stream-processing-hard-problems-part-1-killing-lambda) (Kartik Paramasivam, LinkedIn Engineering 2016)
* [Open Problems in Stream Processing: A Call To Action](https://docs.google.com/presentation/d/1YtTEnOax5MDA8DazDa1ad-sP4zzM58KQK4HNAcxoONA/edit#slide=id.p) (Tyler Akidau, DEBS 2019): Tyler used to lead Dataflow at Google until he joined Snowflake in Jan 2020 to start Snowflake’s streaming team. His talk laid out key challenges of stream processing.
* [The Four Innovation Phases of Netflix's Trillions Scale Real-time Data Infrastructure](https://zhenzhongxu.com/the-four-innovation-phases-of-netflixs-trillions-scale-real-time-data-infrastructure-2370938d7f01) (Zhenzhong Xu, 2022): How Netflix transitioned from a batch system to a streaming system.

## Chapter 4. Training Data

* [Rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling)
* [The MIDAS Touch: Mixed Data Sampling Regression Models](https://escholarship.org/uc/item/9mf223rs) (Ghysels et al., 2004)
* [An Overview of Weak Supervision](https://www.snorkel.org/blog/weak-supervision) (Ratner et al., 2018) 

## Chapter 5. Feature Engineering

* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) (Christoph Molnar, 2022): An amazingly detailed introduction to interpretability

## Chapter 6. Model Development and Offline Evaluation

### Training, debugging, and testing ML code

* [How to unit test machine learning code](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) (Chase Roberts, 2017)
* [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) (Andrej Karpathy, 2019)
* [Top 6 errors novice machine learning engineers make](https://medium.com/ai%C2%B3-theory-practice-business/top-6-errors-novice-machine-learning-engineers-make-e82273d394db) (Christopher Dossman, AI³ | Theory, Practice, Business 2017)
* [Testing and Debugging in Machine Learning](https://developers.google.com/machine-learning/testing-debugging) course (Google)
* [What did you wish you knew before deploying your first ML model?](https://twitter.com/chipro/status/1348265019012743169) (I asked this question on Twitter and got some interesting responses)
* [Techniques for Training Large Neural Networks](https://openai.com/blog/techniques-for-training-large-neural-networks/) (OpenAI 2022) 
* [A survey of model compression and acceleration for deep neural networks](https://arxiv.org/abs/1710.09282) (Cheng et al., IEEE Signal Processing Magazine 2017)
* [Towards Federated Learning at Scale: System Design](https://arxiv.org/abs/1902.01046) (Bonawitz et al, 2019)


### Model evaluation

* [Effective testing for machine learning systems](https://www.jeremyjordan.me/testing-ml/) (Jeremy Jordan, 2020)
* [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (Guo et al., 2017)
* [Calibration for Netflix recommendation systems](https://dl.acm.org/doi/10.1145/3240323.3240372) (Harald Steck, 2018)
* [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/) (Ribeiro et al., ACL 2020)
* [TextBugger: Generating Adversarial Text Against Real-world Applications](https://arxiv.org/abs/1812.05271) (Li et al., 2018) 
* [Uncertainty Sets for Image Classifiers using Conformal Prediction](https://arxiv.org/abs/2009.14193) (Angelopoulos et al., 2020)


## Chapter 7. Model Deployment and Prediction Service


## Chapter 8. Data Distribution Shifts and Monitoring

* [Beyond Incremental Processing: Tracking Concept Drift](https://www.aaai.org/Papers/AAAI/1986/AAAI86-084.pdf) (Jeffrey C. Schlimmer and Richard H. Granger, Jr., 1986). Concept drift isn’t something new!
* [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953) (Rabanser et al., 2019)
* [Out-of-Distribution Generalization via Risk Extrapolation (REx)](http://proceedings.mlr.press/v139/krueger21a.html) (Krueger et al., 2020) 
* [Domain Adaptation under Target and Conditional Shift](https://proceedings.mlr.press/v28/zhang13d.html) (Zhang et al., 2013)
* [A Review of Domain Adaptation without Target Labels](https://ieeexplore.ieee.org/abstract/document/8861136) (Kouw et al., 2019)
* [On Learning Invariant Representations for Domain Adaptation](http://proceedings.mlr.press/v97/zhao19a.html) (Zhao et al., 2019)
* [How to deal with the seasonality of a market?](https://eng.lyft.com/how-to-deal-with-the-seasonality-of-a-market-584cc94d6b75) (Marguerite Graveleau, Lyft Engineering 2019)
* [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893) (Arjovsky et al., 2019) 
* [Causality for Machine Learning](https://arxiv.org/abs/1911.10500) (Bernhard Schölkopf, 2019)


## Chapter 9. Continual Learning and Test in Production

* [Application deployment and testing strategies](https://cloud.google.com/solutions/application-deployment-and-testing-strategies) (Google)
* [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) (Google)
* [Automated Canary Analysis at Netflix with Kayenta](https://netflixtechblog.com/automated-canary-analysis-at-netflix-with-kayenta-3260bc7acc69) (Michael Graff and Chris Sanden, Netflix Technology Blog 2018)


### Contextual bandits

* [A/B testing — Is there a better way? An exploration of multi-armed bandits](https://towardsdatascience.com/a-b-testing-is-there-a-better-way-an-exploration-of-multi-armed-bandits-98ca927b357d) (Greg Rafferty, Towards Data Science 2020)
* [Deep Bayesian Bandits: Exploring in Online Personalized Recommendations](https://arxiv.org/abs/2008.00727) (Guo et al. 2020)
* [Active Learning and Contextual Bandits](http://www.machinedlearnings.com/2012/02/active-learning-and-contextual-bandits.html) (Paul Mineiro, 2012) 


## Chapter 10. Infrastructure and Tooling for MLOps

* [Introduction to Microservices, Docker, and Kubernetes](https://www.youtube.com/watch?v=1xo-0gCVhTU): a good 1-hour video on introduction to Docker and k8s. 
* [How Microsoft plans efficient workloads with DevOps](https://docs.microsoft.com/en-us/azure/devops/learn/devops-at-microsoft/release-flow)
* [Airbnb’s BigHead](https://vimeo.com/274801958)
* [Uber’s Michelangelo](https://eng.uber.com/michelangelo-machine-learning-platform/)


## Chapter 11. The Human Side of Machine Learning

* [Weapons of Math Destruction](https://www.amazon.com/Weapons-Math-Destruction-Increases-Inequality/dp/0553418815) (Cathy O’Neil, Crown Books 2016)
* [NIST Special Publication 1270](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1270.pdf): Towards a Standard for Identifying and Managing Bias in Artificial Intelligence
* ACM Conference on Fairness, Accountability, and Transparency (ACM FAccT) [publications](https://facctconference.org/)
* [Trustworthy ML](https://www.trustworthyml.org/resources)’s recommended list of resources and fundamental papers to researchers and practitioners who want to learn more about trustworthy ML
* Sara Hooker’s awesome slide deck on [ML Beyond Accuracy: Fairness, Security, Governance](https://docs.google.com/presentation/d/1cshMKKSX24L0RL7LNzyOkZNQHD7N-Zyff8iffrLIVYM/edit?usp=sharing) (2022)
* Timnit Gebru and Emily Denton’s [tutorials](https://sites.google.com/view/fatecv-tutorial/schedule) on Fairness, Accountability, Transparency, and Ethics (2020)
