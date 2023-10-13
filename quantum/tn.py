from parseQCP import *
import numpy as np
import os
import time
import sys


class QuantumGates:
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]).reshape(2, 2, 2, 2)
    CX = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ]).reshape(2, 2, 2, 2)
    CZ = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ]).reshape(2, 2, 2, 2)
    CY = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0]
    ]).reshape(2, 2, 2, 2)


def read_states_from_input_file(filename):
    with open(filename, 'r') as file:
        states = [line.strip() for line in file]
    return states


def write_amplitudes_to_output_file(filename, states, amplitudes):
    with open(filename, 'w') as file:
        for state, amplitude in zip(states, amplitudes):
            file.write(f"{state} : {amplitude:.7f}\n")


def write_probabilities_to_output_file(filename, states, probabilities, elapsed_time):
    with open(filename, 'w') as file:
        file.write(f"Wall Clock Time: {elapsed_time:.7f} seconds\n\n")  # Writing the elapsed time
        for state, prob in zip(states, probabilities):
            file.write(f"{state} : {prob:.7f}\n")


class TNtemplate:
    circ = None
    tensor_network: np.ndarray

    def __init__(self, circ: QCPcircuit, truncate=True, error_threshold=0.01) -> None:
        self.circ = circ
        self.truncate = truncate
        self.error_threshold = error_threshold

        self.tensor_network = []
        for i in range(self.circ.numQubits):
            zero_tensor = np.zeros((2, 1, 1), dtype=complex)
            zero_tensor[0, 0, 0] = 1
            self.tensor_network.append(zero_tensor)

    def get_final_state(self):
        state = self.tensor_network[0]
        for tensor in self.tensor_network[1:]:
            state = np.tensordot(state, tensor, axes=([-1], [1]))
        return state.reshape(-1)

    def print_final_state(self):
        state = self.get_final_state()
        for idx, amplitude in enumerate(state):
            if abs(amplitude) > 0.000001:
                print(f"|{format(idx, 'b').zfill(self.circ.numQubits)}> : {amplitude:.4f}")

    def compute_amplitude_for_state(self, state_str):
        amplitude = np.array([[1]])
        for i, tensor in enumerate(self.tensor_network):
            index = int(state_str[i])  # Get the qubit state (0 or 1) from the state string
            amplitude = np.matmul(amplitude, tensor[index])
        return amplitude[0][0]

    def iterate_circ(self):
        if not self.circ:
            raise Exception("circ is None")

        total_gates = len(self.circ.gates)
        start_time = time.time()

        for i, gate in enumerate(self.circ.gates):
            getattr(self, gate.name)(gate)

            current_time = time.time()
            elapsed_time = current_time - start_time

            # Print progress every 60 seconds
            if elapsed_time >= 10:
                print(f"At gate {i + 1}/{total_gates}")
                start_time = current_time  # Reset the timer

    def simulate(self):
        # Iterate Circuit
        self.iterate_circ()

    def x(self, gate: Gate):
        self.apply_single_qubit_gate(gate, QuantumGates.X)

    def y(self, gate: Gate):
        self.apply_single_qubit_gate(gate, QuantumGates.Y)

    def z(self, gate: Gate):
        self.apply_single_qubit_gate(gate, QuantumGates.Z)

    def h(self, gate: Gate):
        self.apply_single_qubit_gate(gate, QuantumGates.H)

    def apply_single_qubit_gate(self, gate: Gate, u_gate: np.ndarray):
        target_tensor = self.tensor_network[gate.target]
        modified_tensor = np.einsum('abc,da->dbc', target_tensor, u_gate)
        self.tensor_network[gate.target] = modified_tensor

    def _apply_two_qubit_gate_logic(self, gate: Gate, u_gate: np.ndarray):
        # Decide the order of tensors and rearrange unitary if needed
        # reversed order
        if gate.control > gate.target:
            qubit0 = self.tensor_network[gate.target]
            qubit1 = self.tensor_network[gate.control]
            u_gate = np.swapaxes(np.swapaxes(u_gate, 0, 1), 2, 3)  # Swap control and target in the unitary
        else:
            qubit0 = self.tensor_network[gate.control]
            qubit1 = self.tensor_network[gate.target]

        # Two Qubit Gate Procedure
        T = np.einsum('abc,dce->adbe', qubit0, qubit1)

        # Contract U and T to T'
        T_strich = np.einsum('abcd,cdef->abef', u_gate, T)

        # Apply SVD to obtain U, S, V^T
        T_strich_reshaped = np.concatenate((np.concatenate((T_strich[0][0], T_strich[0][1]), axis=1),
                                            np.concatenate((T_strich[1][0], T_strich[1][1]), axis=1)), axis=0)
        U, S, V_dagger = np.linalg.svd(T_strich_reshaped, full_matrices=False)

        # Truncation
        if self.truncate:
            # Apply truncation logic if the truncate flag is True
            cumulative_squared_error = 0.0
            for idx, value in enumerate(reversed(S)):
                next_error = cumulative_squared_error + value ** 2
                if next_error > self.error_threshold:
                    chi = max(1, len(S) - idx)  # Ensure at least one value remains
                    break
                cumulative_squared_error = next_error
            else:
                chi = max(1, len(S))  # Ensure at least one value remains
        else:
            # If truncation is disabled, set chi to len(S)
            chi = len(S)

        if chi != len(S):
            # If truncation is necessary, then adjust U, S, and V_dagger
            S_diag = np.diag(S[:chi])
            U = U[:, :chi]
            V_dagger = V_dagger[:chi, :]
        else:
            # If no truncation is needed, just form the diagonal matrix from S
            S_diag = np.diag(S)

        # print(f"Gate {gate.name} ({gate.control}, {gate.target}): Determined chi={chi}, Original S length={len(S)}, Truncated={len(S) - chi}")

        # M = US, M' = V
        M_strich = np.array(np.vsplit(np.matmul(U, S_diag), 2))
        M1_strich = np.array(np.hsplit(V_dagger, 2))

        if gate.control > gate.target:
            self.tensor_network[gate.target], self.tensor_network[gate.control] = M_strich, M1_strich
        else:
            self.tensor_network[gate.control], self.tensor_network[gate.target] = M_strich, M1_strich

    def swap(self, gate):
        self.apply_two_qubit_gate(gate, QuantumGates.SWAP)

    def apply_two_qubit_gate(self, gate, u_gate):
        delta = abs(gate.control - gate.target)
        if delta > 1:
            min_qubit = min(gate.control, gate.target)
            max_qubit = max(gate.control, gate.target)

            for i in range(min_qubit, max_qubit - 1):
                self.swap(Gate("swap", i, i + 1))
                # print(f"Swap: {i} <-> {i + 1}")

            flag_control = 1 if gate.control == max_qubit else 0

            # When I move target from min to max - 1
            if flag_control == 1:
                self._apply_two_qubit_gate_logic(Gate(gate.name, control=max_qubit, target=max_qubit - 1), u_gate)
            # When I move control from min to max - 1
            else:
                self._apply_two_qubit_gate_logic(Gate(gate.name, control=max_qubit - 1, target=max_qubit), u_gate)

            # Bring tensor to original order
            for i in range(max_qubit - 1, min_qubit, -1):
                self.swap(Gate("swap", i, i - 1))
                # print(f"Swap: {i} <-> {i - 1}")
        else:
            self._apply_two_qubit_gate_logic(gate, u_gate)

    def cx(self, gate):
        self.apply_two_qubit_gate(gate, QuantumGates.CX)

    def cz(self, gate):
        self.apply_two_qubit_gate(gate, QuantumGates.CZ)

    def cy(self, gate):
        self.apply_two_qubit_gate(gate, QuantumGates.CY)

    def rx(self, gate):
        angle = gate.param
        angle = angle / 2
        rx_gate = np.array([[np.cos(angle), -1j * np.sin(angle)], [-1j * np.sin(angle), np.cos(angle)]])
        self.apply_single_qubit_gate(gate, rx_gate)

    def ry(self, gate):
        angle = gate.param
        angle = angle / 2
        ry_gate = np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.apply_single_qubit_gate(gate, ry_gate)

    def rz(self, gate):
        angle = gate.param
        angle = angle / 2
        rz_gate = np.array([[np.cos(angle) - 1j * np.sin(angle), 0], [0, np.cos(angle) + 1j * np.sin(angle)]])
        self.apply_single_qubit_gate(gate, rz_gate)

    def measure(self, gate):
        pass


np.set_printoptions(suppress=True)


def main(circuit_name):
    # circuit_name = "ghz_n255"  # Change this variable to the desired circuit name

    # Paths based on circuit_name
    qcp_path = f"challenge/{circuit_name}.qcp"
    input_txt_path = f"challenge/{circuit_name}_input.txt"
    output_txt_path = f"challenge/results/{circuit_name}_output.txt"

    # Ensure results directory exists
    if not os.path.exists("challenge/results/"):
        os.makedirs("challenge/results/")

    # Parse the QCP file
    circuit = parseQCP(qcp_path)

    # Create an instance of your MPS simulator and simulate
    start_time = time.time()
    simulator = TNtemplate(circuit, truncate=True, error_threshold=0.01)
    simulator.simulate()
    elapsed_time = time.time() - start_time

    # Read states from the input file
    states = read_states_from_input_file(input_txt_path)

    # Compute amplitudes & probabilities for the given states
    amplitudes = [simulator.compute_amplitude_for_state(state) for state in states]
    probabilities = [abs(amp) ** 2 for amp in amplitudes]

    # Write amplitudes & probabilities to the output file
    # write_amplitudes_to_output_file(output_txt_path, states, amplitudes)
    write_probabilities_to_output_file(output_txt_path, states, probabilities, elapsed_time)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify a circuit name!")
        sys.exit(1)
    main(sys.argv[1])
