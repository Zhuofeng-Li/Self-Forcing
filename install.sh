#!/bin/bash
# Self-Forcing installation script using uv

# Create virtual environment and install dependencies
echo "Installing dependencies with uv..."
uv sync

# Install flash-attn separately with --no-build-isolation (requires torch to be installed first)
echo "Installing flash-attn..."
uv pip install flash-attn --no-build-isolation

# Install the project in editable mode
echo "Installing project in editable mode..."
uv pip install -e .

echo "Installation complete! Run 'source .venv/bin/activate' to activate the environment."
