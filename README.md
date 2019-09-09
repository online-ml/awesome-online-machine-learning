<div align="center">
    <h1>Awesome Online Machine Learning</h1>
    <a href="https://github.com/sindresorhus/awesome"><img src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg"/></a>
</div>

[Online machine learning](https://www.wikiwand.com/en/Online_machine_learning) is a subset of machine learning where data arrives sequentially. In contrast to the more traditional batch learning, online learning methods update themselves incrementally with one data point at a time.

## Courses and books

- [Machine Learning the Feature](http://www.hunch.net/~mltf/) - Gives some insights into the inner workings of Vowpal Wabbit, especially the [slides on online linear learning](http://www.hunch.net/~mltf/online_linear.pdf).
- [Machine learning for data streams with practical examples in MOA](https://www.cms.waikato.ac.nz/~abifet/book/contents.html)
- [Online Methods in Machine Learning (MIT)](http://www.mit.edu/~rakhlin/6.883/)
- [Streaming 101: The world beyond batch](https://www.oreilly.com/ideas/the-world-beyond-batch-streaming-101)
- [Prediction, Learning, and Games](http://www.ii.uni.wroc.pl/~lukstafi/pmwiki/uploads/AGT/Prediction_Learning_and_Games.pdf)

## Software

- [creme](https://github.com/creme-ml/creme/) - A Python library for general purpose online machine learning.
- [dask](https://ml.dask.org/incremental.html)
- [LIBFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/) - A Library for Field-aware Factorization Machines
- [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) - A Library for Large Linear Classification
- [LIBOL](https://github.com/LIBOL) - A collection of online linear models trained with first and second order gradient descent methods. Not maintained.
- [MOA](https://moa.cms.waikato.ac.nz/documentation/)
- [scikit-learn](https://scikit-learn.org/stable/) - [Some](https://scikit-learn.org/stable/modules/computing.html#incremental-learning) of scikit-learn's estimators can handle incremental updates, although this is usually intended for mini-batch learning.
- [Spark Streaming](https://spark.apache.org/docs/latest/streaming-programming-guide.html) - Doesn't do online learning per say, but instead mini-batches the data into fixed intervals of time.
- [StreamDM](https://github.com/huawei-noah/streamDM) - A machine learning library on top of Spark Streaming.
- [VFML](http://www.cs.washington.edu/dm/vfml/)
- [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit)

## Papers

### Linear models

- [Field-aware Factorization Machines for CTR Prediction (2016)](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [Practical Lessons from Predicting Clicks on Ads atFacebook (2014)](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf)
- [Ad Click Prediction: a View from the Trenches (2013)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
- [Normalized online learning (2013)](https://arxiv.org/abs/1305.6646)
- [Towards Optimal One Pass Large Scale Learning with Averaged Stochastic Gradient Descent (2011)](https://arxiv.org/abs/1107.2490)
- [Dual Averaging Methods for Regularized Stochastic Learning andOnline Optimization (2010)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/xiao10JMLR.pdf)
- [Adaptive Regularization of Weight Vectors (2009)](https://papers.nips.cc/paper/3848-adaptive-regularization-of-weight-vectors.pdf)
- [Stochastic Gradient Descent Training forL1-regularized Log-linear Models with Cumulative Penalty (2009)](https://www.aclweb.org/anthology/P09-1054)
- [Confidence-Weighted Linear Classification (2008)](https://www.cs.jhu.edu/~mdredze/publications/icml_variance.pdf)
- [Exact Convex Confidence-Weighted Learning (2008)](https://www.cs.jhu.edu/~mdredze/publications/cw_nips_08.pdf)
- [Online Passive-Aggressive Algorithms (2006)](http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)
- [A Second-Order Perceptron Algorithm (2005)](http://www.datascienceassn.org/sites/default/files/Second-order%20Perception%20Algorithm.pdf)
- [Online Learning with Kernels (2004)](https://alex.smola.org/papers/2004/KivSmoWil04.pdf)
- [Solving Large Scale Linear Prediction Problems Using Stochastic Gradient Descent Algorithms (2004)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.58.7377)

### Support vector machines

- [Pegasos: Primal Estimated sub-GrAdient SOlver for SVM (2007)](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513)
- [A New Approximate Maximal Margin ClassificationAlgorithm (2001)](http://www.jmlr.org/papers/volume2/gentile01a/gentile01a.pdf)
- [The Relaxed Online Maximum Margin Algorithm (2000)](https://papers.nips.cc/paper/1727-the-relaxed-online-maximum-margin-algorithm.pdf)

### Decision trees

- [AMF: Aggregated Mondrian Forests for Online Learning (2019)](https://arxiv.org/abs/1906.10529)
- [Mondrian Forests: Efficient Online Random Forests (2014)](https://arxiv.org/abs/1406.2673)
- [Mining High-Speed Data Streams (2000)](https://homes.cs.washington.edu/~pedrod/papers/kdd00.pdf)

### Unsupervised learning

- [DeepWalk: Online Learning of Social Representations (2014)](https://arxiv.org/pdf/1403.6652.pdf)
- [Online Learning with Random Representations (2014)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.127.2742&rep=rep1&type=pdf)
- [Online Latent Dirichlet Allocation with Infinite Vocabulary (2013)](http://proceedings.mlr.press/v28/zhai13.pdf)
- [Web-Scale K-Means Clustering (2010)](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)
- [Online Dictionary Learning For Sparse Coding (2009)](https://www.di.ens.fr/sierra/pdfs/icml09.pdf)
- [Density-Based Clustering over an Evolving Data Stream with Noise (2006)](https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf)
- [Knowledge Acquisition Via Incremental Conceptual Clustering (2004)](http://www.inf.ufrgs.br/~engel/data/media/file/Aprendizagem/Cobweb.pdf)
- [Online and Batch Learning of Pseudo-Metrics (2004)](https://ai.stanford.edu/~ang/papers/icml04-onlinemetric.pdf)
- [BIRCH: an efficient data clustering method for very large databases (1996)](https://www2.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf)

### Anomaly detection

- [Fast Anomaly Detection for Streaming Data (2011)](https://www.ijcai.org/Proceedings/11/Papers/254.pdf)

### Ensemble models

- [Optimal and Adaptive Algorithms for Online Boosting (2015)](http://proceedings.mlr.press/v37/beygelzimer15.pdf) - An implementation is available [here](https://github.com/VowpalWabbit/vowpal_wabbit/blob/master/vowpalwabbit/boosting.cc)
- [Online Bagging and Boosting (2001)](https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)
- [A Decision-Theoretic Generalization of On-Line Learningand an Application to Boosting (1997)](http://www.face-rec.org/algorithms/Boosting-Ensemble/decision-theoretic_generalization.pdf)

### Miscellaneous

- [A Complete Recipe for Stochastic Gradient MCMC (2015)](https://arxiv.org/abs/1506.04696)
- [Online EM Algorithm for Latent Data Models (2007)](https://arxiv.org/abs/0712.4273) - Source code is available [here](https://www.di.ens.fr/~cappe/Code/OnlineEM/)

### Surveys

- [Online Learning: A Comprehensive Survey (2018)](https://arxiv.org/abs/1802.02871)
- [Online Machine Learning in Big Data Streams (2018)](https://arxiv.org/abs/1802.05872v1)
- [Incremental Gradient, Subgradient, and Proximal Methods for Convex Optimization: A Survey (2011)](https://arxiv.org/abs/1507.01030)
- [Online Learning and Stochastic Approximations (1998)](https://leon.bottou.org/publications/pdf/online-1998.pdf)

### General-purpose algorithms

- [The Sliding DFT (2003)](https://pdfs.semanticscholar.org/525f/b581f9afe17b6ec21d6cb58ed42d1100943f.pdf) - An online variant of the Fourier Transform, a concise explanation is available [here](https://www.comm.utoronto.ca/~dimitris/ece431/slidingdft.pdf)
- [Maintaining Sliding Window Skylines on Data Streams (2006)](http://www.cs.ust.hk/~dimitris/PAPERS/TKDE06-Sky.pdf)
- [Sketching Algorithms for Big Data](https://www.sketchingbigdata.org/)
