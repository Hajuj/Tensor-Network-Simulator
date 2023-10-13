#!/bin/bash

# List of circuit names
circuit_names=("qft_n4" "qrng_n4" "error_correctiond3_n5" "simon_n6" "rsmvr_n9" "grover_n12" "qf21_n15" "qft_n29" "ghz_n255" "random_entanglement_n30" "vqe_uccsd_n28" "hhl_n14")
total_circuits=${#circuit_names[@]}  # Total number of circuits

echo "Started Simulating $total_circuits Circuits"
echo "-------------------------------------------"
echo ""

# Counter for circuits completed
completed=0

# Loop through each circuit name and call the Python script
for circuit_name in "${circuit_names[@]}"
do
    echo "Started Simulating Circuit $circuit_name"
    python3 quantum/tn.py $circuit_name
    echo "Finished Simulating Circuit $circuit_name"

    completed=$((completed+1))
    echo "Currently done: $completed/$total_circuits Circuits"
    echo ""
done

echo "-------------------------------------------"
echo "Ended Simulating $total_circuits Circuits"
