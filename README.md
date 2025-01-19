#### Introduction

This repo is adapted from my CS671 final project at Duke.

Neural networks excel in tasks like image recognition and optimization, yet their computational demands escalate with higher data dimensions. Quantum computers offer a transformative solution by endowing computations with an exponential advantage in Hilbert Space size. This inherent quantum parallelism holds particular promise for enhancing the capabilities of image recognition models, with Quantum Convolutional Neural Networks (QCNN) emerging as a compelling avenue for advancing classification performance. In my project, I investigate classical-to-quantum data encoding, comparing conventional angle encoding with a novel amplitude encoding scheme. The focus is on binary MNIST classification, aiming to demonstrate successful classification using both encoding methods and contribute insights to quantum machine learning. My work is demonstrated in **QuantumBinaryMNISTClassification** notebook.

I used Xanadu's PennyLane Python package for Quantum Machine Learning (2) for model training, and IBM's Qiskit (3) for visualization tools and work with state vectors.

## References 

1) Deng, L. (2012). The mnist database of handwritten digit images for machine learning research. *IEEE Signal Processing Magazine*, *29*(6), 141–142.
2) Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Carsten Blank, Keri McKiernan and Nathan Killoran. PennyLane. *arXiv*, 2018. arXiv:1811.04968
3) Qiskit contributors. (2023). Qiskit: An Open-source Framework for Quantum Computing.
4) Hoffmann, T., & Brown, D. (2022, November 25). *Gradient estimation with constant scaling for hybrid quantum machine learning*. arXiv.org. https://arxiv.org/abs/2211.13981
5) *MNIST classification: tensorflow quantum*. TensorFlow. https://www.tensorflow.org/quantum/tutorials/mnist?hl=en 
6) Y. Lü, Q. Gao, J. Lü, M. Ogorzałek and J. Zheng, "A Quantum Convolutional Neural Network for Image Classification," 2021 40th Chinese Control Conference (CCC), Shanghai, China, 2021, pp. 6329-6334, doi: 10.23919/CCC52363.2021.9550027.
7) Oh, S., Choi, J., & Kim, J. (2020, September 20). *A tutorial on quantum convolutional Neural Networks (QCNN)*. arXiv.org. https://arxiv.org/abs/2009.09423 
8) Cong, I., Choi, S. & Lukin, M.D. Quantum convolutional neural networks. *Nature Physics* **15**, 1273–1278 (2019). https://doi.org/10.1038/s41567-019-0648-8
9) Bokhan, D., Mastiukova, A. S., Boev, A. S., Trubnikov, D. N., & Fedorov, A. K. (2022, November 27). *Multiclass classification using quantum convolutional neural networks with hybrid quantum-classical learning*. arXiv.org. https://arxiv.org/abs/2203.15368 
10) Hur, T., Kim, L., & Park, D. K. (2022, February 10). *Quantum convolutional neural network for classical data classification - quantum machine intelligence*. SpringerLink. https://link.springer.com/article/10.1007/s42484-021-00061-x 
11) Meuli, G., Soeken, M., Roetteler, M., Bjorner, N., & De Micheli, G. (2019, April 3). *Reversible pebbling game for quantum memory management*. arXiv.org. https://arxiv.org/abs/1904.02121v1 
12) Johnatan Baker. *jmbaker94/quantumcircuitbenchmarks*. GitHub. https://github.com/jmbaker94/quantumcircuitbenchmarks 
13) *Create quantum computing software without limits*. Classiq. https://www.classiq.io/ 
