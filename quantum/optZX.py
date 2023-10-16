import pyzx as zx
import math
from parseQCP import *
import numpy as np


class QuantumCircuitProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.measurement_gates = []
        self.num_qubits = 0
        self.num_gates_before_reduction = 0

    def process_qcp_file(self):
        with open(self.file_path, 'r') as file:
            lines = [line for line in file if not line.strip().startswith('//')]

        self.num_qubits = int(lines[0].strip())
        circuit = zx.Circuit(self.num_qubits)
        self.num_gates_before_reduction = len(lines[1:])

        for line in lines[1:]:
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
                circuit.add_gate("S", int(tokens[2]))
                circuit.add_gate("XPhase", int(tokens[2]))
                circuit.add_gate("S", int(tokens[2]), adjoint=True)
            elif gate == 'rz':
                angle = eval(tokens[1], {'pi': math.pi})
                circuit.add_gate("ZPhase", int(tokens[2]), phase=angle / (2 * math.pi))
            elif gate == 'cx':
                circuit.add_gate("CNOT", int(tokens[1]), int(tokens[2]))
            elif gate == 'cz':
                circuit.add_gate("CZ", int(tokens[1]), int(tokens[2]))
            elif gate == 'cy':
                circuit.add_gate("S", int(tokens[2]), adjoint=True)
                circuit.add_gate("CNOT", int(tokens[1]), int(tokens[2]))
                circuit.add_gate("S", int(tokens[2]))
            elif gate == 'measure':
                gate = Gate('measure')
                gate.target = int(tokens[1])
                self.measurement_gates.append(gate)
            else:
                print(f"Unsupported gate: {gate}")

        zx_graph = circuit.to_graph()

        return zx_graph

    def optimize_zx_graph(self, zx_graph):
        zx.full_reduce(zx_graph)
        # zx.teleport_reduce(zx_graph)
        zx_graph_optimized = zx.extract_circuit(zx_graph)

        return zx_graph_optimized

    def _create_gate(self, gate):
        new_gate = Gate("swap") if gate.name.lower().startswith("swap") else Gate(gate.qasm_name)

        attributes = vars(gate)
        new_gate.target = attributes.get("target", None)
        new_gate.control = attributes.get("control", None)

        if "phase" in attributes:
            new_gate.param = float(attributes["phase"]) * np.pi
        else:
            new_gate.param = None

        return new_gate

    def get_circuit(self, zx_graph_optimized):
        circuit = QCPcircuit()
        circuit.numQubits = int(self.num_qubits)

        gates = [self._create_gate(gate) for gate in zx_graph_optimized.gates]
        num_gates_after_reduction = len(gates)
        circuit.gates = gates + self.measurement_gates
        # print(circuit)
        print("Before: ", self.num_gates_before_reduction - len(self.measurement_gates))
        print("After: ", num_gates_after_reduction)
        print("Percentage reduced: ", (self.num_gates_before_reduction - len(self.measurement_gates)) / num_gates_after_reduction)

        return circuit
