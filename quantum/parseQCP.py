import cmath


class Gate:
    name = None
    param = None
    target = None
    control = None

    def __init__(self, name, control=None, target=None):
        self.name = name
        self.control = control
        self.target = target

    def __repr__(self):
        return self.name

    def __str__(self):
        str = self.name
        if (self.param is not None): str += f"({self.param})"
        if (self.control is not None): str += f" {self.control}"
        if (self.target is not None): str += f" {self.target}"
        return str


class QCPcircuit:
    numQubits = None
    gates = []

    def __str__(self):
        str = f"{self.numQubits}\n"
        str += "\n".join([i.__str__() for i in self.gates])
        return str


def parseQCP(path):
    with open(path, "r") as fp:
        circ = QCPcircuit()
        for line in fp.read().splitlines():
            # ignore comments
            if line.startswith('//'): continue

            # first line that is no comment has to be num of used qbits
            if (circ.numQubits is None):
                circ.numQubits = int(line)
                continue

            gate_comp = line.split()
            gate = Gate(gate_comp[0])

            # gates with parameters
            if (line.startswith('r')):
                gate.param = float(eval(gate_comp[1].replace("pi", str(cmath.pi))))
                gate.target = int(gate_comp[2])
                circ.gates.append(gate)
                continue

            # controlled gates
            if (line.startswith('c')):
                gate.control = int(gate_comp[1])
                gate.target = int(gate_comp[2])
                circ.gates.append(gate)
                continue

            # single qbit gates without parameters
            gate.target = int(gate_comp[1])
            circ.gates.append(gate)
    return circ


if __name__ == "__main__":
    c = parseQCP("QCPBench/small/grover_n2.qcp")
    print(c)
