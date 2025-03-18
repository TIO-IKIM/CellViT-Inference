#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Remove old build if it exists
echo "Removing old build..."
rm -rf ./cellvit.egg-info ./dist

# Run the build command
echo "Running build..."
python -m build

echo "Build completed successfully."
