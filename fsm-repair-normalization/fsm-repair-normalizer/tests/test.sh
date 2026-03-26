#!/bin/bash
set -e

pytest /tests/test_outputs.py

# REQUIRED: reward file
mkdir -p /logs/verifier
echo 1 > /logs/verifier/reward.txt
