## Introduction

This repo is adapted from my CS671 final project at Duke.

Neural networks excel in tasks like image recognition and optimization, yet their computational demands escalate with higher data dimensions. Quantum computers offer a transformative solution by endowing computations with an exponential advantage in Hilbert Space size. This inherent quantum parallelism holds particular promise for enhancing the capabilities of image recognition models, with Quantum Convolutional Neural Networks (QCNN) emerging as a compelling avenue for advancing classification performance. In my project, I investigate classical-to-quantum data encoding, comparing conventional angle encoding with a novel amplitude encoding scheme. The focus is on binary MNIST classification, aiming to demonstrate successful classification using both encoding methods and contribute insights to quantum machine learning. My work is demonstrated in **QuantumBinaryMNISTClassification** notebook.

I used Xanadu's PennyLane Python package for Quantum Machine Learning (2) for model training, and IBM's Qiskit (3) for visualization tools and work with state vectors.

## Method

In the upcoming exploration, two distinct data encoding methods will be investigated: conventional angle encoding and amplitude encoding. In the conventional angle encoding approach, the requisite number of qubits equals the number of pixels in the image, denoted as N. Each pixel's intensity is encoded through a rotation that is proportionate to its intensity value, facilitating straightforward implementation on a quantum backend. However, it's important to note that for larger images, such as 28x28, this method necessitates a quantum system comprising 784 qubits. This renders simulation unfeasible on classical devices and poses challenges for training on current noisy quantum hardware.

In the amplitude encoding method, the encoding scheme is more intricate. The n$^{\text{th}}$ training sample, denoted as $X_n$, can be encoded as $\ket{X_n}=\frac{1}{\sqrt{N}}\sum_{i=0}^{N-1}\ket{i}\otimes\ket{I_i}$. Here, the i$^{\text{th}}$ term in the $\ket{X_n}$ superposition corresponds to the pixel's number i in binary representation, tensor-multiplied with its intensity $I_i$. Consequently, the number of qubits required to encode an image with N pixels is given by $ceil(\log(N))+1=O(\log(N))$, where we need $ceil(\log(N))$ qubits to track N numbers and an additional qubit for intensity representation. Notably, this method utilizes logarithmically fewer qubits compared to angle encoding. However, it comes at the expense of requiring arbitrary state preparation, which can be an operation of exponential cost (9).

For angle encoding-based classification, I adopted the tensor network approach employing four qubits, as outlined in the PennyLane demo (https://pennylane.ai/qml/demos/tutorial_tn_circuits/). Each two-qubit block in this configuration incorporates two RY gates and a CNOT, and each subsequent layer utilizes half the number of qubits compared to the preceding one. The final qubit in the network measures the expectation value of the Pauli $\hat{Z}$ matrix, denoted as $\braket{\hat{Z}}$. A negative $\braket{\hat{Z}}$ is interpreted as the model predicting the presence of a horizontal line in the image. Furthermore, during the exploration of 4x4 lines classification, an alternative block architecture will be investigated.

## Results

As mentioned in the introduction, the goal was to classify compressed to 4x4 size 3s and 6s from MNIST handwritten digits datasets. The total number of training images is 12049 and the total number of testing images is 1968. I was training my models on only 1000 images as recommended in (4).

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
