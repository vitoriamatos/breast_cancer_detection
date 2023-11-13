#!/bin/bash

# Set the root directory
root_dir="./"

# Create databases directories
mkdir -p "$root_dir/databases/mammotherm_database"
mkdir -p "$root_dir/databases/mias_database"

# Change to the root directory
cd "$root_dir"

# Create methrics directories
mkdir -p "methrics/mammotherm/tests"
mkdir -p "methrics/mias/tests"

# Change to the root directory
cd "$root_dir"

# Create models directories
mkdir -p "models/mammotherm/tests"
mkdir -p "models/mias/tests"

# Change to the root directory
cd "$root_dir"

# Install requirements
# pip install -r requirements.txt
