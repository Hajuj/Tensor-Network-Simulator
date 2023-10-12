from parseQCP import *
import numpy as np


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
        if (self.circ is None): raise Exception("circ is None")
        for gate in self.circ.gates:
            print("Tensornetwork before Gate: ", self.tensor_network)
            getattr(self, gate.name)(gate)
            print("Gate: ", gate.name, " applied!")
            print("Tensornetwork after Gate: ", self.tensor_network)

    def simulate(self):
        # Iterate Circuit
        self.iterate_circ()

    def x(self, gate: Gate):
        x_gate = np.array([[0, 1], [1, 0]])
        self.apply_single_qubit_gate(gate, x_gate)

    def y(self, gate: Gate):
        y_gate = np.array([[0, -1j], [1j, 0]])
        self.apply_single_qubit_gate(gate, y_gate)

    def z(self, gate: Gate):
        z_gate = np.array([[1, 0], [0, -1]])
        self.apply_single_qubit_gate(gate, z_gate)

    def h(self, gate: Gate):
        h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self.apply_single_qubit_gate(gate, h_gate)

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
            print("Have to Swap!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            qubit0 = self.tensor_network[gate.control]
            qubit1 = self.tensor_network[gate.target]
            print("No Swap!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Two Qubit Gate Procedure
        print("Control: ", qubit0)
        print("Target: ", qubit1)
        T = np.einsum('abc,dce->adbe', qubit0, qubit1)
        print("Contracted 4 dim T: ", T)

        ## Contract U and T to T'
        T_strich = np.einsum('abcd,cdef->abef', u_gate, T)
        print("Contracted 4 dim UT aka. T': ", T_strich)

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

        print("M_strich: ", M_strich)
        print("M1_strich: ", M1_strich)

        if gate.control > gate.target:
            self.tensor_network[gate.target], self.tensor_network[gate.control] = M_strich, M1_strich
        else:
            self.tensor_network[gate.control], self.tensor_network[gate.target] = M_strich, M1_strich

        print("Result: ", M_strich[1] @ M1_strich[1])

    def swap(self, gate):
        U_swap = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]).reshape(2, 2, 2, 2)
        self.apply_two_qubit_gate(gate, U_swap)

    def apply_two_qubit_gate(self, gate, u_gate):
        delta = abs(gate.control - gate.target)
        if delta > 1:
            min_qubit = min(gate.control, gate.target)
            max_qubit = max(gate.control, gate.target)

            for i in range(min_qubit, max_qubit - 1):
                self.swap(Gate("swap", i, i + 1))
                print("Swap in the one direction, time: ", i)

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
                print("Swap in the one direction, time: ", i)
        else:
            self._apply_two_qubit_gate_logic(gate, u_gate)
            print("Apply Gate: ", gate.name)

    def cx(self, gate):
        U_cx = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]).reshape(2, 2, 2, 2)
        self.apply_two_qubit_gate(gate, U_cx)

    def cz(self, gate):
        U_cz = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ]).reshape(2, 2, 2, 2)
        self.apply_two_qubit_gate(gate, U_cz)

    def cy(self, gate):
        U_cy = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ]).reshape(2, 2, 2, 2)
        self.apply_two_qubit_gate(gate, U_cy)

    def rx(self, gate):
        angle = gate.param
        angle = angle / 2
        rx_gate = np.array([[np.cos(angle), -1j * np.sin(angle)], [-1j * np.sin(angle), np.cos(angle)]])
        target_tensor = self.tensor_network[gate.target]
        modified_tensor = np.einsum('abc,dc->abd', target_tensor, rx_gate)
        self.tensor_network[gate.target] = modified_tensor

    def ry(self, gate):
        angle = gate.param
        angle = angle / 2
        ry_gate = np.array([[np.cos(angle), -1 * np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target_tensor = self.tensor_network[gate.target]
        modified_tensor = np.einsum('abc,dc->abd', target_tensor, ry_gate)
        self.tensor_network[gate.target] = modified_tensor

    def rz(self, gate):
        angle = gate.param
        angle = angle / 2
        rz_gate = np.array([[np.cos(angle) - 1j * np.sin(angle), 0], [0, -np.cos(angle) + 1j * np.sin(angle)]])
        target_tensor = self.tensor_network[gate.target]
        modified_tensor = np.einsum('abc,dc->abd', target_tensor, rz_gate)
        self.tensor_network[gate.target] = modified_tensor

    def measure(self, gate):
        m0 = np.array([[1, 0], [0, 0]])
        m1 = np.array([[0, 0], [0, 1]])
        m0_dagger = np.conj(m0).T
        M_0 = np.matmul(m0_dagger, m0)

        identity = np.identity(2)
        identities = [identity for _ in range(self.circ.numQubits)]
        identities[gate.target] = M_0
        unitary = 1
        for identity in identities:
            unitary = np.kron(identity, unitary)
        tensor_network_complex = np.conj(self.tensor_network).T
        p_0 = np.matmul(np.matmul(tensor_network_complex, unitary), self.tensor_network)

        random_choice = np.random.choice([0, 1], p=[p_0, (1 - p_0)])
        M_m = m0 if random_choice == 0 else m1

        numerator = np.matmul(M_m, self.tensor_network)
        denominator = np.sqrt(np.matmul(np.matmul(tensor_network_complex, M_m), self.tensor_network))

        result = numerator / denominator
        self.tensor_network = np.matmul(result, self.tensor_network)


np.set_printoptions(suppress=True)

if __name__ == "__main__":
    # c = parseQCP("code/QCPBench/small/test_n1.qcp")
    c = parseQCP("../QCPBench/small/test_n1.qcp")
    simulator = TNtemplate(c)
    simulator.simulate()
