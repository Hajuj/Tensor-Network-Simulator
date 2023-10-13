import os
import pyzx as zx
import math
import matplotlib.pyplot as plt


class QuantumCircuitProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def process_qcp_file(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        num_qubits = int(lines[1].strip())
        circuit = zx.Circuit(num_qubits)

        for line in lines[2:]:
            tokens = line.strip().split()
            gate = tokens[0].lower()

            if gate == 'x':
                circuit.add_gate("NOT", int(tokens[1]))
            elif gate == 'y':
                circuit.add_gate("S", int(tokens[1]))
                circuit.add_gate("NOT", int(tokens[1]))
                circuit.add_gate("S", int(tokens[1]), adjoint=True)
            elif gate == 'z':
                circuit.add_gate("Z", int(tokens[1]))
            elif gate == 'h':
                circuit.add_gate("HAD", int(tokens[1]))
            elif gate == 'rx':
                angle = eval(tokens[1], {'pi': math.pi})
                circuit.add_gate("XPhase", int(tokens[2]), phase=angle / (2 * math.pi))
            elif gate == 'ry':
                # Implementing RY gate using RX and RZ
                angle = eval(tokens[1], {'pi': math.pi})
                circuit.add_gate("RZ", int(tokens[2]), phase=angle / (2 * math.pi))
                circuit.add_gate("NOT", int(tokens[2]))
                circuit.add_gate("RZ", int(tokens[2]), phase=-angle / (2 * math.pi))
                circuit.add_gate("NOT", int(tokens[2]))
            elif gate == 'rz':
                angle = eval(tokens[1], {'pi': math.pi})
                circuit.add_gate("ZPhase", int(tokens[2]), phase=angle / (2 * math.pi))
            elif gate == 'cx':
                circuit.add_gate("CNOT", int(tokens[1]), int(tokens[2]))
            elif gate == 'cz':
                circuit.add_gate("CZ", int(tokens[1]), int(tokens[2]))
            elif gate == 'cy':
                circuit.add_gate("S", int(tokens[2]))
                circuit.add_gate("CNOT", int(tokens[1]), int(tokens[2]))
                circuit.add_gate("S", int(tokens[2]), adjoint=True)
            elif gate == 'measure':
                # PyZX doesn't have a direct measure gate
                pass
            else:
                print(f"Unsupported gate: {gate}")

        return circuit


def main():
    # Path to the 'challenge' directory
    input_directory = os.path.join(os.pardir, 'challenge')
    output_directory = os.path.join(input_directory, 'optimized')
    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".qcp"):
            file_path = os.path.join(input_directory, file_name)
            processor = QuantumCircuitProcessor(file_path)
            circuit = processor.process_qcp_file()

            # Print the name of the file being processed
            print(f"Processing File: {file_name}")

            # Print the circuit to the console
            print(circuit)

            # Visualize the circuit (optional)
            try:
                zx.draw(circuit)
                plt.show()
            except ImportError:
                print("Matplotlib not installed. Cannot visualize the circuit.")

            # Apply PyZX optimization here
            # For example: zx.full_reduce(circuit.to_graph())
            # And then convert it back to a circuit if needed

            processor.save_optimized_circuit(circuit, output_directory)


if __name__ == "__main__":
    main()
