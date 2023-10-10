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
      self.chi = 30

  def iterate_circ(self):
    if (self.circ is None): raise Exception("circ is None") 
    for gate in self.circ.gates:
      print("Tensornetwork: ", self.tensor_network)
      getattr(self, gate.name)(gate)
      print("Gate: ", gate.name, " applied!")
      print("Tensornetwork: ", self.tensor_network)

  def simulate(self):
    # TODO 
    # Iterate Circuit
    self.iterate_circ()

  def x(self, gate: Gate):
    x_gate = np.array([[0, 1], [1, 0]])
    target_tensor = self.tensor_network[gate.target]
    modified_tensor = np.einsum('abc,da->dbc', target_tensor, x_gate)
    self.tensor_network[gate.target] = modified_tensor

  def y(self, gate):
    y_gate = np.array([[0, -1j], [1j, 0]])
    target_tensor = self.tensor_network[gate.target]
    modified_tensor = np.einsum('abc,da->dbc', target_tensor, y_gate)
    self.tensor_network[gate.target] = modified_tensor

  def z(self, gate):
    z_gate = np.array([[1, 0], [0, -1]])
    target_tensor = self.tensor_network[gate.target]
    modified_tensor = np.einsum('abc,da->dbc', target_tensor, z_gate)
    self.tensor_network[gate.target] = modified_tensor

  def h(self, gate):
    h_gate = np.array([[1, 1], [1, -1]])
    h_gate = h_gate / np.sqrt(2)
    target_tensor = self.tensor_network[gate.target]
    modified_tensor = np.einsum('abc,da->dbc', target_tensor, h_gate)
    self.tensor_network[gate.target] = modified_tensor

  def cx(self, gate):
    # Setup
    # zero = np.array([[1, 0], [0, 0]])
    # one = np.array([[0, 0], [0, 1]])
    # x_gate = np.array([[0, 1], [1, 0]])
    # identity = np.identity(2)

    # identities = [identity for _ in range(self.circ.numQubits)]
    # identities_x = [identity for _ in range(self.circ.numQubits)]

    # identities[gate.control] = zero
    # identities_x[gate.control] = one
    # identities_x[gate.target] = x_gate

    # unitary_1 = unitary_2 = 1
    # for iden, iden_x in zip(identities, identities_x):
    #     unitary_1 = np.kron(iden, unitary_1)
    #     unitary_2 = np.kron(iden_x, unitary_2)

    # Apply
    # unitary = unitary_1 + unitary_2
    # self.tensor_network = np.matmul(unitary, self.tensor_network)

    ## Setup
    control_tensor = self.tensor_network[gate.control]
    target_tensor = self.tensor_network[gate.target]
    U_gate = np.array([[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]])
    
    ## Contract M(n) & M(n+1) to T
    print("Control: ", control_tensor)
    print("Target: ", target_tensor)
    T = np.einsum('abc,dce->adbe', control_tensor, target_tensor)
    print("Contracted 4 dim T: ", T)

    ## Contract U and T to T'
    T_strich = np.einsum('abcd,cdef->abef', U_gate, T)
    print("Contracted 4 dim UT aka. T': ", T_strich)

    ## Apply SVD to obtain U, S, V^T
    E, F, A, B = T_strich.shape
    T_strich_reshaped = T_strich.reshape((E * A, F * B))
    print(T_strich_reshaped)
    U, S, V_dagger = np.linalg.svd(T_strich_reshaped, full_matrices=False)
    #V = np.conj(V_dagger).T ## check if it works
    print("S: ", S)
    # S_diag = np.diag(S[:self.chi])  # chi is the bond dimension you want to keep

    # ## check truncation
    # U_truncated = U[:, self.chi]
    # S_truncated = S[self.chi]
    # V_truncated = V_dagger[self.chi, :]

    
    # Multiply singular values into U and V
    M_strich = np.array(np.vsplit(np.matmul(U, np.diag(S)), 2)) #vsplit
    M1_strich = np.array(np.hsplit(V_dagger, 2)) #hsplit

    print("M_strich: ", M_strich)
    print("M1_strich: ", M1_strich)

    self.tensor_network[gate.control] = M_strich
    self.tensor_network[gate.target] = M1_strich
    
    print("Result: ", M_strich[1] @ M1_strich[1])
    

    
  def cz(self, gate):
    # Setup
    zero = np.array([[1, 0], [0, 0]])
    one = np.array([[0, 0], [0, 1]])
    z_gate = np.array([[0, 1], [1, 0]])
    identity = np.identity(2)

    identities = [identity for _ in range(self.circ.numQubits)]
    identities_z = [identity for _ in range(self.circ.numQubits)]

    identities[gate.control] = zero
    identities_z[gate.control] = one
    identities_z[gate.target] = z_gate

    unitary_1 = unitary_2 = 1
    for iden, iden_z in zip(identities, identities_z):
        unitary_1 = np.kron(iden, unitary_1)
        unitary_2 = np.kron(iden_z, unitary_2)
    
    # Apply
    unitary = unitary_1 + unitary_2
    self.tensor_network = np.matmul(unitary, self.tensor_network)

  def cy(self, gate):
    # Setup
    zero = np.array([[1, 0], [0, 0]])
    one = np.array([[0, 0], [0, 1]])
    y_gate = np.array([[0, 1], [1, 0]])
    identity = np.identity(2)

    identities = [identity for _ in range(self.circ.numQubits)]
    identities_y = [identity for _ in range(self.circ.numQubits)]

    identities[gate.control] = zero
    identities_y[gate.control] = one
    identities_y[gate.target] = y_gate

    unitary_1 = unitary_2 = 1
    for iden, iden_y in zip(identities, identities_y):
        unitary_1 = np.kron(iden, unitary_1)
        unitary_2 = np.kron(iden_y, unitary_2)
    
    # Apply
    unitary = unitary_1 + unitary_2
    self.tensor_network = np.matmul(unitary, self.tensor_network)


  def rx(self, gate):
    angle = gate.param
    angle = angle/2
    rx_gate = np.array([[np.cos(angle), -1j*np.sin(angle)], [-1j*np.sin(angle), np.cos(angle)]])
    target_tensor = self.tensor_network[gate.target]
    modified_tensor = np.einsum('abc,dc->abd', target_tensor, rx_gate)
    self.tensor_network[gate.target] = modified_tensor


  def ry(self, gate):
    angle = gate.param
    angle = angle/2
    ry_gate = np.array([[np.cos(angle), -1*np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    target_tensor = self.tensor_network[gate.target]
    modified_tensor = np.einsum('abc,dc->abd', target_tensor, ry_gate)
    self.tensor_network[gate.target] = modified_tensor

  def rz(self, gate):
    angle = gate.param
    angle = angle/2
    rz_gate = np.array([[np.cos(angle)-1j*np.sin(angle), 0], [0, -np.cos(angle)+ 1j*np.sin(angle)]])
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

    random_choice = np.random.choice([0, 1], p=[p_0, (1-p_0)])
    M_m = m0 if random_choice == 0 else m1
    
    numerator = np.matmul(M_m, self.tensor_network)
    denominator = np.sqrt(np.matmul(np.matmul(tensor_network_complex, M_m), self.tensor_network))

    result = numerator/denominator
    self.tensor_network = np.matmul(result, self.tensor_network)




if __name__ == "__main__":
  #c = parseQCP("code/QCPBench/small/test_n1.qcp")
  c = parseQCP("QCPBench/small/test_n1.qcp")
  simulator = TNtemplate(c)
  simulator.simulate()
