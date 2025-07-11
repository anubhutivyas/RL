# Set Up Clusters

This guide explains how to run NeMo RL with Ray on Slurm or Kubernetes.

## Use Slurm for Batched and Interactive Jobs

 The following code provides instructions on how to use Slurm to run batched job submissions and run jobs interactively.

### Batched Job Submission

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

COMMAND="uv run ./examples/run_grpo_math.py" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```

:::{tip}
Depending on your Slurm cluster configuration, you may or may not need to include the `--gres=gpu:8` option in the `sbatch` command.
:::

Upon successful submission, Slurm will print the `SLURM_JOB_ID`:
```text
Submitted batch job 1980204
```
Make a note of the job submission number. Once the job begins, you can track its process in the driver logs which you can `tail`:
```sh
tail -f 1980204-logs/ray-driver.log
```

### Interactive Launching

:::{tip}
A key advantage of running interactively on the head node is the ability to execute multiple multi-node jobs without needing to requeue in the Slurm job queue. This means that during debugging sessions, you can avoid submitting a new `sbatch` command each time. Instead, you can debug and re-submit your NeMo RL job directly from the interactive session.
:::

To run interactively, launch the same command as [Batched Job Submission](#batched-job-submission), but omit the `COMMAND` line:
```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=1  # Total nodes requested (head is colocated on ray-worker-0)

CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=1:0:0 \
    --gres=gpu:8 \
    ray.sub
```
Upon successful submission, Slurm will print the `SLURM_JOB_ID`:
```text
Submitted batch job 1980204
```
Once the Ray cluster is up, a script will be created to attach to the Ray head node. Run this script to launch experiments:
```sh
bash 1980204-attach.sh
```
Now that you are on the head node, you can launch the command as follows:
```sh
uv run ./examples/run_grpo_math.py
```

### Slurm Environment Variables

All Slurm environment variables described below can be added to the `sbatch`
invocation of `ray.sub`. For example, `GPUS_PER_NODE=8` can be specified as follows:

```sh
GPUS_PER_NODE=8 \
... \
sbatch ray.sub \
   ...
```
#### Common Environment Configuration
``````{list-table}
:header-rows: 1

* - Environment Variable
  - Explanation
* - `CONTAINER`
  - (Required) Specifies the container image to be used for the Ray cluster.
    Use either a docker image from a registry or a squashfs (if using enroot/pyxis).
* - `MOUNTS`
  - (Required) Defines paths to mount into the container. Examples:
    ```md
    * `MOUNTS="$PWD:$PWD"` (mount in current working directory (CWD))
    * `MOUNTS="$PWD:$PWD,/nfs:/nfs:ro"` (mounts the current working directory and `/nfs`, with `/nfs` mounted as read-only)
    ```
* - `COMMAND`
  - Command to execute after the Ray cluster starts. If empty, the cluster idles and enters interactive mode (see the [Slurm interactive instructions](#interactive-launching)).
* - `HF_HOME`
  - Sets the cache directory for huggingface-hub assets (e.g., models/tokenizers).
* - `WANDB_API_KEY`
  - Setting this allows you to use the wandb logger without having to run `wandb login`.
* - `HF_TOKEN`
  - Setting the token used by huggingface-hub. Avoids having to run the `huggingface-cli login`
* - `HF_DATASETS_CACHE`
  - Sets the cache dir for downloaded Huggingface datasets.
``````

:::{tip}
When `HF_TOKEN`, `WANDB_API_KEY`, `HF_HOME`, and `HF_DATASETS_CACHE` are set in your shell environment using `export`, they are automatically passed to `ray.sub`. For instance, if you set:

```sh
export HF_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```
this token will be available to your NeMo RL run. Consider adding these exports to your shell configuration file, such as `~/.bashrc`.
:::

#### Advanced Environment Configuration
``````{list-table}
:header-rows: 1

* - Environment Variable
    (and default)
  - Explanation
* - `UV_CACHE_DIR_OVERRIDE`
  - By default, this variable does not need to be set. If unset, `ray.sub` uses the 
    `UV_CACHE_DIR` defined within the container (defaulting to `/root/.cache/uv`). 
    `ray.sub` intentionally avoids using the `UV_CACHE_DIR` from the user's host 
    environment to prevent the host's cache from interfering with the container's cache. 
    Set `UV_CACHE_DIR_OVERRIDE` if you have a customized `uv` environment (e.g., 
    with pre-downloaded packages or specific configurations) that you want to persist 
    and reuse across container runs. This variable should point to a path on a shared 
    filesystem accessible by all nodes (head and workers). This path will be mounted 
    into the container and will override the container's default `UV_CACHE_DIR`.
* - `CPUS_PER_WORKER=128`
  - CPUs each Ray worker node claims. Default is `16 * GPUS_PER_NODE`.
* - `GPUS_PER_NODE=8`
  - Number of GPUs each Ray worker node claims. To determine this, run `nvidia-smi` on a worker node.
* - `BASE_LOG_DIR=$SLURM_SUBMIT_DIR`
  - Base directory for storing Ray logs. Defaults to the Slurm submission directory ([SLURM_SUBMIT_DIR](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_SUBMIT_DIR)).
* - `NODE_MANAGER_PORT=53001`
  - Port for the Ray node manager on worker nodes.
* - `OBJECT_MANAGER_PORT=53003`
  - Port for the Ray object manager on worker nodes.
* - `RUNTIME_ENV_AGENT_PORT=53005`
  - Port for the Ray runtime environment agent on worker nodes.
* - `DASHBOARD_AGENT_GRPC_PORT=53007`
  - gRPC port for the Ray dashboard agent on worker nodes.
* - `METRICS_EXPORT_PORT=53009`
  - Port for exporting metrics from worker nodes.
* - `PORT=6379`
  - Main port for the Ray head node.
* - `RAY_CLIENT_SERVER_PORT=10001`
  - Port for the Ray client server on the head node.
* - `DASHBOARD_GRPC_PORT=52367`
  - gRPC port for the Ray dashboard on the head node.
* - `DASHBOARD_PORT=8265`
  - Port for the Ray dashboard UI on the head node. This is also the port
    used by the Ray distributed debugger.
* - `DASHBOARD_AGENT_LISTEN_PORT=52365`
  - Listening port for the dashboard agent on the head node.
* - `MIN_WORKER_PORT=54001`
  - Minimum port in the range for Ray worker processes.
* - `MAX_WORKER_PORT=54257`
  - Maximum port in the range for Ray worker processes.
``````

:::{note}
For the most part, you will not need to change ports unless these
are already taken by some other service backgrounded on your cluster.
:::

## Kubernetes

NeMo RL can be deployed on Kubernetes clusters for scalable, containerized training. This section covers setting up Ray clusters on Kubernetes and running NeMo RL jobs.

### Prerequisites

Before deploying NeMo RL on Kubernetes, ensure you have:

- **Kubernetes Cluster**: A functional Kubernetes cluster with GPU nodes
- **NVIDIA GPU Operator**: Installed and configured for GPU access
- **Helm**: For managing Ray cluster deployments
- **kubectl**: Configured to access your Kubernetes cluster
- **Container Registry Access**: Access to pull NeMo RL container images

### Deploy Ray Cluster

Use the Ray Helm chart to deploy a Ray cluster on Kubernetes:

```bash
# Add the Ray Helm repository
helm repo add ray https://ray-project.github.io/ray-charts/
helm repo update

# Create a values file for your Ray cluster
cat > ray-cluster-values.yaml << EOF
ray:
  head:
    replicas: 1
    resources:
      limits:
        cpu: "4"
        memory: "8Gi"
        nvidia.com/gpu: 1
      requests:
        cpu: "2"
        memory: "4Gi"
        nvidia.com/gpu: 1
    container:
      image: nvcr.io/nvidia/pytorch:23.12-py3
      command: ["ray", "start", "--head", "--port=6379"]
  
  worker:
    replicas: 3  # Adjust based on your needs
    resources:
      limits:
        cpu: "8"
        memory: "32Gi"
        nvidia.com/gpu: 8
      requests:
        cpu: "4"
        memory: "16Gi"
        nvidia.com/gpu: 8
    container:
      image: nvcr.io/nvidia/pytorch:23.12-py3
      command: ["ray", "start", "--address=\$RAY_HEAD_SERVICE_HOST:6379"]

  dashboard:
    enabled: true
    port: 8265
EOF

# Deploy the Ray cluster
helm install ray-cluster ray/ray-cluster -f ray-cluster-values.yaml
```

### Verify Cluster Status

Check that your Ray cluster is running properly:

```bash
# Check Ray cluster pods
kubectl get pods -l ray.io/cluster=ray-cluster

# Check Ray cluster services
kubectl get services -l ray.io/cluster=ray-cluster

# Access Ray dashboard
kubectl port-forward svc/ray-cluster-head-svc 8265:8265
```

### Run NeMo RL Jobs

Once your Ray cluster is running, you can submit NeMo RL jobs:

```bash
# Create a job configuration
cat > nemo-rl-job.yaml << EOF
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: nemo-rl-training
spec:
  entrypoint: |
    #!/bin/bash
    git clone https://github.com/NVIDIA/NeMo-RL.git
    cd NeMo-RL
    pip install uv
    uv run ./examples/run_grpo_math.py
  runtimeEnv:
    pip:
      - torch>=2.0.0
      - transformers>=4.30.0
      - accelerate
      - ray[default]
  clusterSpec:
    headGroupSpec:
      serviceType: ClusterIP
      replicas: 1
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
          - name: ray-head
            image: nvcr.io/nvidia/pytorch:23.12-py3
            resources:
              limits:
                nvidia.com/gpu: 1
              requests:
                nvidia.com/gpu: 1
    workerGroupSpecs:
    - replicas: 3
      rayStartParams: {}
      template:
        spec:
          containers:
          - name: ray-worker
            image: nvcr.io/nvidia/pytorch:23.12-py3
            resources:
              limits:
                nvidia.com/gpu: 8
              requests:
                nvidia.com/gpu: 8
EOF

# Submit the job
kubectl apply -f nemo-rl-job.yaml
```

### Monitor Jobs

Monitor your NeMo RL training jobs:

```bash
# Check job status
kubectl get rayjobs

# View job logs
kubectl logs -f rayjob/nemo-rl-training

# Access Ray dashboard for job monitoring
kubectl port-forward svc/ray-cluster-head-svc 8265:8265
```

### Environment Configuration

Set up environment variables for your Kubernetes deployment:

```bash
# Create a ConfigMap for environment variables
cat > nemo-rl-config.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: nemo-rl-config
data:
  HF_HOME: "/workspace/cache"
  WANDB_API_KEY: "your_wandb_api_key"
  HF_TOKEN: "your_hf_token"
  HF_DATASETS_CACHE: "/workspace/datasets"
EOF

kubectl apply -f nemo-rl-config.yaml
```

### Persistent Storage

For long-running training jobs, configure persistent storage:

```yaml
# Add to your Ray cluster values
ray:
  head:
    volumeMounts:
    - name: cache-storage
      mountPath: /workspace/cache
    - name: datasets-storage
      mountPath: /workspace/datasets
    volumes:
    - name: cache-storage
      persistentVolumeClaim:
        claimName: nemo-rl-cache-pvc
    - name: datasets-storage
      persistentVolumeClaim:
        claimName: nemo-rl-datasets-pvc
```

### Scaling Considerations

- **GPU Resources**: Ensure your Kubernetes nodes have sufficient GPU resources
- **Network Bandwidth**: High-speed interconnects improve distributed training performance
- **Storage I/O**: Use high-performance storage for dataset caching and checkpointing
- **Memory Requirements**: Monitor memory usage and adjust resource limits accordingly

### Troubleshooting

Common issues and solutions:

```bash
# Check GPU availability
kubectl get nodes -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# Verify Ray cluster connectivity
kubectl exec -it ray-cluster-head-0 -- ray status

# Check resource usage
kubectl top pods -l ray.io/cluster=ray-cluster
```

For more detailed Kubernetes deployment options, refer to the [Ray Kubernetes documentation](https://docs.ray.io/en/latest/ray-core/configure.html#kubernetes).
