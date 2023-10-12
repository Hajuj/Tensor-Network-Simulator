from parseQCP import *
import numpy as np

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

class TNtemplate:
    circ = None
    tensor_network: np.ndarray
    chi: int

    def __init__(self, circ: QCPcircuit) -> None:
        self.circ = circ
        self.tensor_network = []
        for i in range(self.circ.numQubits):
            zero_tensor = np.zeros((2, 1, 1), dtype=complex)
            zero_tensor[0, 0, 0] = 1
            self.tensor_network.append(zero_tensor)
            self.chi = 2

    def iterate_circ(self):
      if not self.circ:
          raise Exception("circ is None")
      print("Initial Tensor:", self.tensor_network)
      for gate in self.circ.gates:
          getattr(self, gate.name)(gate)
          print(f"Gate {gate.name} applied!")
          print("Tensornetwork after Gate:", self.tensor_network)

      print("\nFINAL MPS")
      for i, tensor in enumerate(self.tensor_network):
          print(f"Qubit {i} Tensor is:\n{tensor}\n")

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
        ## reversed order
        if gate.control > gate.target:
            qubit0 = self.tensor_network[gate.target]
            qubit1 = self.tensor_network[gate.control]
            u_gate = np.swapaxes(np.swapaxes(u_gate, 0, 1), 2, 3)  # Swap control and target in the unitary
        else:
            qubit0 = self.tensor_network[gate.control]
            qubit1 = self.tensor_network[gate.target]

        # Two Qubit Gate Procedure
        T = np.einsum('abc,dce->adbe', qubit0, qubit1)

        ## Contract U and T to T'
        T_strich = np.einsum('abcd,cdef->abef', u_gate, T)

        ## Apply SVD to obtain U, S, V^T
        T_strich_reshaped = np.concatenate((np.concatenate((T_strich[0][0], T_strich[0][1]), axis=1),
                                            np.concatenate((T_strich[1][0], T_strich[1][1]), axis=1)), axis=0)
        U, S, V_dagger = np.linalg.svd(T_strich_reshaped, full_matrices=False)

        chi_min = min(self.chi, len(S))  ## error checking for chi TODO: Make this more sophisticated
        S_diag = np.diag(S[:self.chi])
        U = U[:, :chi_min]
        V_dagger = V_dagger[:chi_min, :]

        print("Truncation to: ", chi_min, " from ", len(S),
              " and S had shape: ", S.shape, " while S_diag trunc. has: ", S_diag.shape)

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
                print(f"Swap: {i} <-> {i + 1}")

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
                print(f"Swap: {i} <-> {i - 1}")
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
        rz_gate = np.array([[np.cos(angle) - 1j * np.sin(angle), 0], [0, -np.cos(angle) + 1j * np.sin(angle)]])
        self.apply_single_qubit_gate(gate, rz_gate)

    def measure(self, gate):
        pass

np.set_printoptions(suppress=True)

if __name__ == "__main__":
    # c = parseQCP("code/QCPBench/small/test_n1.qcp")
    c = parseQCP("QCPBench/small/test_n1.qcp")
    simulator = TNtemplate(c)
    simulator.simulate()
