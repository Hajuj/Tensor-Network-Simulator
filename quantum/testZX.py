import pyzx as zx
import math
from parseQCP import *


class ZX:
    circ = None

    def __init__(self, circ: QCPcircuit) -> None:
        self.circ = circ

    # Function to parse and process the .qcp file
    def process_qcp_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Create a PyZX circuit
        num_qubits = int(lines[1].strip())
        circuit = zx.Circuit(self.circ.numQubits)

        # Process each gate operation
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
                angle = eval(tokens[1])  # Be careful with eval, ensure the input is safe
                circuit.add_gate("XPhase", int(tokens[2]), phase=angle / (2 * math.pi))
            elif gate == 'ry':
                # Implementing RY gate using RX and RZ
                angle = eval(tokens[1])
                circuit.add_gate("RZ", int(tokens[2]), phase=angle / (2 * math.pi))
                circuit.add_gate("NOT", int(tokens[2]))
                circuit.add_gate("RZ", int(tokens[2]), phase=-angle / (2 * math.pi))
                circuit.add_gate("NOT", int(tokens[2]))
            elif gate == 'rz':
                angle = eval(tokens[1])
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
                # PyZX doesn't have a direct measure gate, handle according to your needs
                pass
            else:
                print(f"Unsupported gate: {gate}")

        return circuit


if __name__ == "__main__":
    # Example usage
    file_path = parseQCP("../QCPBench/small/test_n1.qcp")  # Replace with the actual path to your .qcp file
    circuit = ZX(file_path)
    print(circuit)
