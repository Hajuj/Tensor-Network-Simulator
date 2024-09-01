# Tensor Network Simulator for Quantum Circuits

This project implements a Tensor Network Simulator designed to simulate quantum circuits efficiently. The simulator leverages Matrix Product States (MPS) as the underlying tensor network architecture, allowing for effective simulation of quantum systems with a large number of qubits while managing the complexity associated with entanglement.

## Key Features

- **Matrix Product States (MPS):** The simulator uses MPS to represent quantum states. This approach is advantageous for handling systems where the complexity primarily stems from the degree of entanglement rather than the sheer number of qubits.
  
- **SVD Truncation:** To optimize the simulation, Singular Value Decomposition (SVD) is employed to truncate the tensors during the contraction process. This helps keep the tensor dimensions manageable, thus improving computational efficiency without significant loss of accuracy.
  
- **ZX Calculus for Circuit Optimization:** The project also integrates circuit optimization techniques using ZX Calculus. ZX Calculus is a graphical language for quantum circuits that facilitates simplifying and optimizing quantum circuits before simulation.

## Optimization Techniques

1. **SVD Truncation:**
   - After applying two-qubit gates, the simulator uses SVD to decompose the resulting tensors and truncates small singular values. This reduces the rank of the tensors and limits the growth of bond dimensions, enabling efficient simulation of deep quantum circuits.

2. **Circuit Optimization Using ZX Calculus:**
   - The ZX Calculus method optimizes the quantum circuits before simulation. By simplifying the circuit, the number of gates and entanglements is reduced, directly impacting the simulation process's efficiency and accuracy.

## Applications

This Tensor Network Simulator is particularly useful for simulating quantum circuits in scenarios where traditional methods face scalability issues. It can be applied to various quantum algorithms and research experiments, providing insights into the behavior of quantum systems with practical computational resources.


## Installation
Please run the following command to install the necessary dependencies:
```pip install -r requirements.txt```

## Link to Pitch.com
[Presentation](https://pitch.com/public/fa739879-e5bb-4fa7-830a-f78a8b119b7f)

## Authors and acknowledgment
Christian, Mohamad

This project is part of the practical course "Quantum Computing" offered by LMU Munich.
