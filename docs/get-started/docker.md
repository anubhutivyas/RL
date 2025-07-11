# Docker Setup

This guide provides three methods for building Docker images:

* **release**: Contains everything from the hermetic image, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers.
* **hermetic**: Includes the base image plus pre-fetched NeMo RL python packages in the `uv` cache.
* **base**: A minimal image with CUDA, `ray`, and `uv` installed, ideal for specifying Python dependencies at runtime.

Use the:
* **release** (recommended): if you want to pre-fetch the NeMo RL [worker virtual environments](local-workstation) and copy in the project source code.
* **hermetic**: if you want to pre-fetch NeMo RL python packages into the `