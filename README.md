## Introduction

This repo is adapted from my Fall 2023 CS671 final project at Duke.

Neural networks excel in tasks like image recognition and optimization, yet their computational demands escalate with higher data dimensions. Quantum computers offer a transformative solution by endowing computations with an exponential advantage in Hilbert Space size. This inherent quantum parallelism holds particular promise for enhancing the capabilities of image recognition models, with Quantum Convolutional Neural Networks (QCNN) emerging as a compelling avenue for advancing classification performance. In my project, I investigate classical-to-quantum data encoding, comparing conventional angle encoding with a novel amplitude encoding scheme. The focus is on binary MNIST classification, aiming to demonstrate successful classification using both encoding methods and contribute insights to quantum machine learning. My work is demonstrated in **QuantumBinaryMNISTClassification** notebook.

I used Xanadu's PennyLane Python package for Quantum Machine Learning (2) for model training, and IBM's Qiskit (3) for visualization tools and work with state vectors.

## Method

In the upcoming exploration, two distinct data encoding methods will be investigated: conventional angle encoding and amplitude encoding. In the conventional angle encoding approach, the requisite number of qubits equals the number of pixels in the image, denoted as N. Each pixel's intensity is encoded through a rotation that is proportionate to its intensity value, facilitating straightforward implementation on a quantum backend. However, it's important to note that for larger images, such as 28x28, this method necessitates a quantum system comprising 784 qubits. This renders simulation unfeasible on classical devices and poses challenges for training on current noisy quantum hardware.

In the amplitude encoding method, the encoding scheme is more intricate. The n'th training sample, denoted as \( X_n \), can be encoded as 

$$
\ket{X_n} = \frac{1}{\sqrt{N}} \sum_{i=0}^{N-1} \ket{i} \otimes \ket{I_i}
$$

Here, the i'th term in the $\ket{X_n}$ superposition corresponds to the pixel's number i in binary representation, tensor-multiplied with its intensity $I_i$. Consequently, the number of qubits required to encode an image with N pixels is given by $ceil(\log(N))+1=O(\log(N))$, where we need $ceil(\log(N))$ qubits to track N numbers and an additional qubit for intensity representation. Notably, this method utilizes logarithmically fewer qubits compared to angle encoding. However, it comes at the expense of requiring arbitrary state preparation, which can be an operation of exponential cost (9).

For angle encoding-based classification, I adopted the tensor network approach employing four qubits, as outlined in the PennyLane demo (https://pennylane.ai/qml/demos/tutorial_tn_circuits/). Each two-qubit block in this configuration incorporates two RY gates and a CNOT, and each subsequent layer utilizes half the number of qubits compared to the preceding one. The final qubit in the network measures the expectation value of the Pauli $\hat{Z}$ matrix, denoted as $\braket{\hat{Z}}$. A negative $\braket{\hat{Z}}$ is interpreted as the model predicting the presence of a horizontal line in the image. Furthermore, during the exploration of 4x4 lines classification, an alternative block architecture will be investigated.

The objective function is chosen to be the same as in the PennyLane demo:

$$\mathcal{L}(\theta)=-\sum_{i=1}^{N}\hat{y_i}\text{sign}(\hat{y_i})$$,

where $\hat{y}\in[-1,1]$ is the model's prediction for i'th image. Thus, if $\mathcal{L}(\theta)=-N$, model $\theta$ confidently classify all N training samples. By "confidently" I mean assigning labels strictly -1 and 1 to the right samples.

The full quantum circuit with angle encoding is presented in the **Angle Encoding** section of the notebook. The first layer of RY gates is responsible for data encoding and is followed by three convolutional blocks with a total of six parameters to be learned. The full quantum circuit with amplitude encoding is presented in the **Amplitude Encoding** section of the notebook. State preparation module, $\ket{\Psi}$, is followed by two layers with a total of six parameters to be learned.

## Results

As mentioned in the introduction, the goal was to classify compressed to 4x4 size 3s and 6s from MNIST handwritten digits datasets. The total number of training images is 12049 and the total number of testing images is 1968. I was training my models on only 1000 images as recommended in (4).

he models' performance is summarized in the following table:

|                                       | Angle Encoding, Accuracy | Amplitude Encoding, Accuracy | Angle Encoding, F1 Score | Amplitude Encoding, F1 Score |
| :-----------------------------------: | :----------------------: | :--------------------------: | :----------------------: | :--------------------------: |
|     Training Data  (1000 images)      |          0.803           |            0.874             |          0.829           |            0.869             |
|      Testing Data (1968 images)       |          0.780           |            0.891             |          0.803           |            0.883             |
| Original Training Set  (12049 images) |          0.777           |            0.866             |          0.803           |            0.857             |

The amplitude encoding model works better!!!

## Discussion

1. **Results**: In this project, I executed binary MNIST classification using angle encoding and amplitude encoding schemes. Notably, the amplitude encoding model exhibited superior performance, achieving higher accuracy (0.891 vs 0.780) and F1 score (0.883 vs 0.803) compared to its angle encoding counterpart. These findings underscore the potential efficacy of amplitude encoding in quantum machine learning tasks.
2. **Qubit Number - Dimensionality Tradeoff**: Quantum computers derive their advantage from the expansive Hilbert space they cover. However, there is a tradeoff inherent in the proposed amplitude encoding method. While successful in this study, using fewer qubits corresponds to a smaller Hilbert space. In certain cases, this reduction may lead to the absence of a classification hyperplane within the limited Hilbert space, potentially limiting the model's effectiveness.
3. **Multiqubit Gates**: In selecting an ansatz for the amplitude encoding circuit, a key observation emerged regarding the efficacy of multi-controlled X (MCX) gates. A crucial insight revolves around entanglement in quantum circuits. Firstly, the entangling scheme, utilizing N (N-1)-controlled X gates, proves advantageous only when a basis state with (N-1) 1s exists in the initial superposition. Otherwise, the MCX gates will not exert influence. To fully harness quantum entanglement, an exploration of $N\choose i$ i-controlled gates for the N-qubit circuit is proposed (acknowledging the associated computational cost). Secondly, for ansatz optimization, examining the resultant N-qubit state vector is recommended. If the state closely approximates a product state $\bigotimes_{i=1}^{N}\ket{\psi_i}$, suggesting limited entanglement, the number of multi-qubit gates in the ansatz can be judiciously reduced, facilitating computational speedup.
4. **Running on Quantum Hardware**: While my primary objective did not entail creating a circuit trainable on near-term quantum devices, the exploration focused on atypical data encoding. Notably, the circuit involved costly operations such as arbitrary state preparation and multi-controlled X (MCX) gates in terms of circuit depth. However, there are effective strategies to mitigate these challenges. The pebbling strategy (11), as implemented in the Quantum Circuit Benchmarks software (12), adeptly decomposes MCX gates into multiple Toffoli gates, leveraging ancilla qubits. Additionally, tools like Classiq (13) prove efficient in performing state preparation with specified error bounds. 
5. **Training Speed**: Simulating the evolution of the state vector poses computational challenges, reflecting in the time required for training. In the case of MNIST binary classification, utilizing a 16-qubit angle encoding circuit demanded approximately five hours to learn thirty parameters. Conversely, training a 5-qubit amplitude encoding circuit with ten parameters was accomplished in a more time-efficient manner, requiring two hours. It is noteworthy that for amplitude encoding, PennyLane's StatePrep block, which "attempts to decompose the operation" (https://docs.pennylane.ai/en/latest/code/api/pennylane.StatePrep.html), was employed. Given the absence of the goal to execute circuits on quantum backends, exploration of a new hardware-agnostic state preparation method is warranted to enhance training efficiency.
6. **Further Work**: The aim is to achieve over 90% accuracy in binary MNIST classification using the angle encoding scheme, with specific insights discussed in the Errors section. Shifting towards multiclass classification, the focus will remain on the amplitude encoding method. For a four-class scenario, the plan is to measure the expectation value on two qubits and assign classes based on the signs of the measured values ([-1,-1] - class 1; [-1, 1] - class 2; [1,-1] - class 3; [1,1] - class 4). Future steps also involve exploring larger image sizes and experimenting with different entangling schemes to refine the model's performance.


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
