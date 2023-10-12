import os
import re

# Returns a list of all .qcp circuit names in sorted order depending on the number of qubits involved
def get_circuit_names(directory):
    # List all files in the given directory
    files = os.listdir(directory)
    
    # Filter out files that are not .qcp
    qcp_files = [f for f in files if f.endswith('.qcp')]
    
    # Extract the names without the .qcp extension
    circuit_names = [os.path.splitext(f)[0] for f in qcp_files]

    # Extract qubit count from the circuit names using regex and sort based on it
    circuit_names = sorted(circuit_names, key=lambda name: int(re.search(r'n(\d+)', name).group(1)))
    
    return circuit_names

if __name__ == "__main__":
    circuit_names = get_circuit_names("challenge")
    print(" ".join(['"' + name + '"' for name in circuit_names]))
