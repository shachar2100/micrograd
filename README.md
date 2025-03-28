# micrograd: A Minimal Autograd Engine for Deep Learning Education

**micrograd** is a tiny educational library created by Andrej Karpathy to illustrate the inner workings of neural network training. As explained in his lecture "[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4178s)", this library provides a step-by-step understanding of how backpropagation and automatic gradient calculation (autograd) are implemented.

## What is micrograd?

micrograd is a **scalar-valued autograd engine**. Autograd, short for automatic gradient, is the core mechanism behind training modern deep neural networks. micrograd specifically implements **backpropagation**, an algorithm that efficiently calculates the gradient of a loss function with respect to the neural network's weights. This gradient information allows us to iteratively adjust the weights to minimize the loss and improve the network's accuracy.

## Why is micrograd interesting?

*   **Pedagogical Clarity:** micrograd is designed for **educational purposes**. It breaks down the complexities of neural network training to its fundamental scalar operations (individual numbers and basic arithmetic like plus and times). This avoids the initial complexity of multi-dimensional tensors used in production libraries, making the underlying math and algorithms easier to grasp.
*   **Minimal Codebase:** The core autograd engine of micrograd is implemented in **approximately 100 lines of simple Python code** in `engine.py`. The neural network library built on top (`nn.py`) is also very concise, around 50 lines. This small size allows for a complete understanding of the entire process.
*   **Illustrates Fundamental Concepts:** By building micrograd from scratch, you can gain an intuitive understanding of:
    *   **Derivatives:** What they represent and how they relate to the sensitivity of a function's output to its inputs.
    *   **Chain Rule:** How gradients are propagated backward through a computation graph.
    *   **Forward Pass:** Building a mathematical expression and calculating its output.
    *   **Backward Pass:** Computing gradients of the output with respect to all intermediate variables and inputs using backpropagation.
    *   **Gradient Descent:** Using gradient information to iteratively update parameters (weights and biases) to minimize a loss function.
*   **Foundation for Larger Libraries:** micrograd demonstrates the fundamental principles that power larger, production-ready deep learning libraries like **PyTorch** and **JAX**. Understanding micrograd provides a solid foundation for learning and using these more complex tools.

## How does micrograd work?

1.  **Value Object:** micrograd introduces a `Value` object that wraps scalar numbers. These `Value` objects not only store data but also keep track of the operations they are involved in (their "children" and the operation that created them).
2.  **Building Expressions:** You can build mathematical expressions by performing operations (+, \*, power, negation, etc.) on these `Value` objects. micrograd automatically constructs a **computation graph** representing these operations and dependencies.
3.  **Forward Pass:** When you compute the final output of an expression, micrograd performs a forward pass through the graph, calculating the value at each node. The output of the forward pass can be accessed using the `.data` attribute.
4.  **Backward Pass (Backpropagation):** By calling the `.backward()` method on the final output `Value`, micrograd initiates backpropagation. This process traverses the computation graph in reverse, recursively applying the chain rule of calculus to calculate the gradient of the output with respect to every `Value` object in the graph. Each `Value` object stores its gradient in the `.grad` attribute. The derivative of a node with respect to the output (loss) is initially zero, and the `.backward()` method initializes the gradient of the final output to 1.
5.  **Neural Network Library (`nn.py`):** Built on top of the `Value` object and the autograd engine, `nn.py` provides basic building blocks for neural networks:
    *   **Neuron:** A fundamental unit that takes multiple inputs, applies weights and a bias, and passes the result through a non-linear activation function (like tanh or ReLU). The weights and bias are initialized randomly.
    *   **Layer:** A collection of independent neurons.
    *   **MLP (Multi-Layer Perceptron):** A sequence of interconnected layers of neurons.
6.  **Training:** micrograd enables the training of these neural networks by:
    *   Defining a **loss function** that measures the discrepancy between the network's predictions and the desired targets. The goal of training is to minimize this loss.
    *   Performing a forward pass to get the predictions and calculate the loss.
    *   Performing a backward pass on the loss to compute gradients with respect to all the network's parameters (weights and biases). It's important to **zero the gradients** before each backward pass to prevent accumulation from previous passes.
    *   Updating the parameters using **gradient descent** (or similar optimization algorithms) to minimize the loss. This involves nudging the parameters in the opposite direction of their gradient.

## Repository Structure

*   `engine.py`: Contains the implementation of the `Value` object and the core autograd engine (backpropagation).
*   `nn.py`: Implements the neural network building blocks (`Module`, `Neuron`, `Layer`, `MLP`) on top of the `engine.py`.
*   `test.py`: Includes tests to verify the correctness of micrograd by comparing its forward and backward passes with PyTorch.
*   `demo.ipynb`: A Jupyter Notebook demonstrating a more complex binary classification example using micrograd.

## Getting Started

To start exploring micrograd, you can:

1.  Clone this repository.
2.  Open and run the `demo.ipynb` notebook in a Jupyter environment.
3.  Examine the `engine.py` and `nn.py` files to understand the implementation details.
4.  Refer to Andrej Karpathy's lecture "[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3erMg0Q)" for a detailed explanation and walkthrough.

## Relation to PyTorch

As demonstrated in the lecture and the `test.py` file, micrograd's design and functionality closely mirror the fundamental concepts of PyTorch. While PyTorch is significantly more efficient and feature-rich (handling tensors, supporting various hardware like GPUs, offering a wide range of neural network layers and optimization algorithms), micrograd provides a simplified, scalar-based perspective on the core mechanisms. Understanding micrograd can greatly enhance your comprehension of how PyTorch and other deep learning frameworks operate under the hood.

## Further Learning and Discussion

For further discussion and questions related to micrograd, please refer to the discussion forum or group linked in the video description of the associated lecture.

## Contributions

While this is primarily an educational project, contributions and feedback are welcome.

## License

`MIT License`
