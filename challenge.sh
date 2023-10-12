#!/bin/bash

# List of circuit names
circuit_names=("qft_n4" "grover_n12" "ghz_n255")
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
