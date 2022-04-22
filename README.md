<div align="center">
    <h1>Awesome Online Machine Learning</h1>
    <a href="https://github.com/sindresorhus/awesome"><img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg"/></a>
</div>

[Online machine learning](https://www.wikiwand.com/en/Online_machine_learning) is a subset of machine learning where data arrives sequentially. In contrast to the more traditional batch learning, online learning methods update themselves incrementally with one data point at a time.

- [Courses and books](#courses-and-books)
- [Blog posts](#blog-posts)
- [Software](#software)
  - [Modelling](#modelling)
  - [Deployment](#deployment)
- [Papers](#papers)
  - [Linear models](#linear-models)
  - [Support vector machines](#support-vector-machines)
  - [Neural networks](#neural-networks)
  - [Decision trees](#decision-trees)
  - [Unsupervised learning](#unsupervised-learning)
  - [Time series](#time-series)
  - [Drift detection](#drift-detection)
  - [Anomaly detection](#anomaly-detection)
  - [Metric learning](#metric-learning)
  - [Graph theory](#graph-theory)
  - [Ensemble models](#ensemble-models)
  - [Expert learning](#expert-learning)
  - [Miscellaneous](#miscellaneous)
  - [Surveys](#surveys)
  - [General-purpose algorithms](#general-purpose-algorithms)
  - [Hyperparameter tuning](#hyperparameter-tuning)

## Courses and books

- [IE 498: Online Learning and Decision Making](https://yuanz.web.illinois.edu/teaching/IE498fa19/)
- [Introduction to Online Learning](https://parameterfree.com/lecture-notes-on-online-learning/)
- [Machine Learning the Feature](http://www.hunch.net/~mltf/) — Gives some insights into the inner workings of Vowpal Wabbit, especially the [slides on online linear learning](http://www.hunch.net/~mltf/online_linear.pdf).
- [Machine learning for data streams with practical examples in MOA](https://www.cms.waikato.ac.nz/~abifet/book/contents.html)
- [Online Methods in Machine Learning (MIT)](http://www.mit.edu/~rakhlin/6.883/)
- [Streaming 101: The world beyond batch](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-101)
- [Prediction, Learning, and Games](http://www.ii.uni.wroc.pl/~lukstafi/pmwiki/uploads/AGT/Prediction_Learning_and_Games.pdf)
- [Introduction to Online Convex Optimization](https://ocobook.cs.princeton.edu/OCObook.pdf)
- [Reinforcement Learning and Stochastic Optimization: A unified framework for sequential decisions](https://castlelab.princeton.edu/RLSO/) — The entire book builds upon Online Learning paradigm in applied learning/optimization problems, *Chapter 3  Online learning* being the reference.

## Blog posts

- [Anomaly Detection with Bytewax & Redpanda (Bytewax, 2022)](https://www.bytewax.io/blog/anomaly-detection-bw-rpk/)
- [The online machine learning predict/fit switcheroo (Max Halford, 2022)](https://maxhalford.github.io/blog/predict-fit-switcheroo/)
- [Real-time machine learning: challenges and solutions (Chip Huyen, 2022)](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html)
- [Anomalies detection using River (Matias Aravena Gamboa, 2021)](https://medium.com/spikelab/anomalies-detection-using-river-398544d3536)
- [Introdução (não-extensiva) a Online Machine Learning (Saulo Mastelini, 2021)](https://medium.com/@saulomastelini/introdu%C3%A7%C3%A3o-a-online-machine-learning-874bd6b7c3c8)
- [Machine learning is going real-time (Chip Huyen, 2020)](https://huyenchip.com/2020/12/27/real-time-machine-learning.html)
- [The correct way to evaluate online machine learning models (Max Halford, 2020)](https://maxhalford.github.io/blog/online-learning-evaluation/)
- [What is online machine learning? (Max Pagels, 2018)](https://medium.com/value-stream-design/online-machine-learning-515556ff72c5)
- [What Is It and Who Needs It (Data Science Central, 2015)](https://www.datasciencecentral.com/profiles/blogs/stream-processing-what-is-it-and-who-needs-it)

## Software

### Modelling

- [River](https://github.com/creme-ml/creme/) — A Python library for general purpose online machine learning.
- [dask](https://ml.dask.org/incremental.html)
- [Jubatus](http://jubat.us/en/index.html)
- [LIBFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/) — A Library for Field-aware Factorization Machines
- [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) — A Library for Large Linear Classification
- [LIBOL](https://github.com/LIBOL) — A collection of online linear models trained with first and second order gradient descent methods. Not maintained.
- [MOA](https://moa.cms.waikato.ac.nz/documentation/)
- [scikit-learn](https://scikit-learn.org/stable/) — [Some](https://scikit-learn.org/stable/modules/computing.html#incremental-learning) of scikit-learn's estimators can handle incremental updates, although this is usually intended for mini-batch learning. See also the ["Computing with scikit-learn"](https://scikit-learn.org/stable/modules/computing.html) page.
- [Spark Streaming](https://spark.apache.org/docs/latest/streaming-programming-guide.html) — Doesn't do online learning per say, but instead mini-batches the data into fixed intervals of time.
- [SofiaML](https://code.google.com/archive/p/sofia-ml/)
- [StreamDM](https://github.com/huawei-noah/streamDM) — A machine learning library on top of Spark Streaming.
- [Tornado](https://github.com/alipsgh/tornado)
- [VFML](http://www.cs.washington.edu/dm/vfml/)
- [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit)

### Deployment

- [KappaML](https://www.kappaml.com/)
- [django-river-ml](https://github.com/vsoch/django-river-ml) — a Django plugin for deploying River models
- [chantilly](https://github.com/online-ml/chantilly) — a prototype meant to be compatible with River (previously Creme)

## Papers

### Linear models

- [Field-aware Factorization Machines for CTR Prediction (2016)](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [Practical Lessons from Predicting Clicks on Ads at Facebook (2014)](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf)
- [Ad Click Prediction: a View from the Trenches (2013)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
- [Normalized online learning (2013)](https://arxiv.org/abs/1305.6646)
- [Towards Optimal One Pass Large Scale Learning with Averaged Stochastic Gradient Descent (2011)](https://arxiv.org/abs/1107.2490)
- [Dual Averaging Methods for Regularized Stochastic Learning andOnline Optimization (2010)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/xiao10JMLR.pdf)
- [Adaptive Regularization of Weight Vectors (2009)](https://papers.nips.cc/paper/3848-adaptive-regularization-of-weight-vectors.pdf)
- [Stochastic Gradient Descent Training forL1-regularized Log-linear Models with Cumulative Penalty (2009)](https://www.aclweb.org/anthology/P09-1054)
- [Confidence-Weighted Linear Classification (2008)](https://www.cs.jhu.edu/~mdredze/publications/icml_variance.pdf)
- [Exact Convex Confidence-Weighted Learning (2008)](https://www.cs.jhu.edu/~mdredze/publications/cw_nips_08.pdf)
- [Online Passive-Aggressive Algorithms (2006)](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
- [Logarithmic Regret Algorithms forOnline Convex Optimization (2007)](https://www.cs.princeton.edu/~ehazan/papers/log-journal.pdf)
- [A Second-Order Perceptron Algorithm (2005)](http://www.datascienceassn.org/sites/default/files/Second-order%20Perception%20Algorithm.pdf)
- [Online Learning with Kernels (2004)](https://alex.smola.org/papers/2004/KivSmoWil04.pdf)
- [Solving Large Scale Linear Prediction Problems Using Stochastic Gradient Descent Algorithms (2004)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.58.7377)

### Support vector machines

- [Pegasos: Primal Estimated sub-GrAdient SOlver for SVM (2007)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513)
- [A New Approximate Maximal Margin Classification Algorithm (2001)](http://www.jmlr.org/papers/volume2/gentile01a/gentile01a.pdf)
- [The Relaxed Online Maximum Margin Algorithm (2000)](https://papers.nips.cc/paper/1727-the-relaxed-online-maximum-margin-algorithm.pdf)

### Neural networks

- [Three scenarios for continual learning (2019)](https://arxiv.org/pdf/1904.07734.pdf)

### Decision trees

- [AMF: Aggregated Mondrian Forests for Online Learning (2019)](https://arxiv.org/abs/1906.10529)
- [Mondrian Forests: Efficient Online Random Forests (2014)](https://arxiv.org/abs/1406.2673)
- [Mining High-Speed Data Streams (2000)](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf)

### Unsupervised learning

- [Online hierarchical clustering approximations (2019)](https://arxiv.org/pdf/1909.09667.pdf)
- [DeepWalk: Online Learning of Social Representations (2014)](https://arxiv.org/pdf/1403.6652.pdf)
- [Online Learning with Random Representations (2014)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.127.2742&rep=rep1&type=pdf)
- [Online Latent Dirichlet Allocation with Infinite Vocabulary (2013)](http://proceedings.mlr.press/v28/zhai13.pdf)
- [Web-Scale K-Means Clustering (2010)](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)
- [Online Dictionary Learning For Sparse Coding (2009)](https://www.di.ens.fr/sierra/pdfs/icml09.pdf)
- [Density-Based Clustering over an Evolving Data Stream with Noise (2006)](https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf)
- [Knowledge Acquisition Via Incremental Conceptual Clustering (2004)](http://www.inf.ufrgs.br/~engel/data/media/file/Aprendizagem/Cobweb.pdf)
- [Online and Batch Learning of Pseudo-Metrics (2004)](https://ai.stanford.edu/~ang/papers/icml04-onlinemetric.pdf)
- [BIRCH: an efficient data clustering method for very large databases (1996)](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf)

### Time series

- [Online Learning for Time Series Prediction (2013)](https://arxiv.org/pdf/1302.6927.pdf)

### Drift detection

- [A Survey on Concept Drift Adaptation (2014)](http://eprints.bournemouth.ac.uk/22491/1/ACM%20computing%20surveys.pdf)

### Anomaly detection

- [Leveraging the Christoffel-Darboux Kernel for Online Outlier Detection (2022)](https://hal.laas.fr/hal-03562614/document)
- [Interpretable Anomaly Detection with Mondrian Pólya Forests on Data Streams (2020)](https://arxiv.org/pdf/2008.01505.pdf)
- [Fast Anomaly Detection for Streaming Data (2011)](https://www.ijcai.org/Proceedings/11/Papers/254.pdf)

### Metric learning

- [Online Metric Learning and Fast Similarity Search (2009)](http://people.bu.edu/bkulis/pubs/nips_online.pdf)
- [Information-Theoretic Metric Learning (2007)](http://www.cs.utexas.edu/users/pjain/pubs/metriclearning_icml.pdf)
- [Online and Batch Learning of Pseudo-Metrics (2004)](https://ai.stanford.edu/~ang/papers/icml04-onlinemetric.pdf)

### Graph theory

- [DeepWalk: Online Learning of Social Representations (2014)](http://www.cs.cornell.edu/courses/cs6241/2019sp/readings/Perozzi-2014-DeepWalk.pdf)

### Ensemble models

- [Optimal and Adaptive Algorithms for Online Boosting (2015)](http://proceedings.mlr.press/v37/beygelzimer15.pdf) — An implementation is available [here](https://github.com/VowpalWabbit/vowpal_wabbit/blob/master/vowpalwabbit/boosting.cc)
- [Online Bagging and Boosting (2001)](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)
- [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting (1997)](http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf)

### Expert learning

- [On the optimality of the Hedge algorithm in the stochastic regime](https://arxiv.org/pdf/1809.01382.pdf)

### Miscellaneous

- [Multi-Output Chain Models and their Application in Data Streams (2019)](https://jmread.github.io/talks/2019_03_08-Imperial_Stats_Seminar.pdf)
- [A Complete Recipe for Stochastic Gradient MCMC (2015)](https://arxiv.org/abs/1506.04696)
- [Online EM Algorithm for Latent Data Models (2007)](https://arxiv.org/abs/0712.4273) — Source code is available [here](https://www.di.ens.fr/~cappe/Code/OnlineEM/)

### Surveys

- [Online Learning: A Comprehensive Survey (2018)](https://arxiv.org/abs/1802.02871)
- [Online Machine Learning in Big Data Streams (2018)](https://arxiv.org/abs/1802.05872v1)
- [Incremental learning algorithms and applications (2016)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2016-19.pdf)
- [Batch-Incremental versus Instance-Incremental Learning in Dynamic and Evolving Data](http://albertbifet.com/wp-content/uploads/2013/10/IDA2012.pdf)
- [Incremental Gradient, Subgradient, and Proximal Methods for Convex Optimization: A Survey (2011)](https://arxiv.org/abs/1507.01030)
- [Online Learning and Stochastic Approximations (1998)](https://leon.bottou.org/publications/pdf/online-1998.pdf)

### General-purpose algorithms

- [Maintaining Sliding Window Skylines on Data Streams (2006)](http://www.cs.ust.hk/~dimitris/PAPERS/TKDE06-Sky.pdf)
- [The Sliding DFT (2003)](https://pdfs.semanticscholar.org/525f/b581f9afe17b6ec21d6cb58ed42d1100943f.pdf) — An online variant of the Fourier Transform, a concise explanation is available [here](https://www.comm.utoronto.ca/~dimitris/ece431/slidingdft.pdf)
- [Sketching Algorithms for Big Data](https://www.sketchingbigdata.org/)

### Hyperparameter tuning

- [ChaCha for Online AutoML (2021)](https://arxiv.org/pdf/2106.04815.pdf)
