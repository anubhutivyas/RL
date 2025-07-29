#!/usr/bin/env python3
"""Training Loop with Dynamic Autoscaling.

This script simulates a training loop that:
1. Connects to existing Ray cluster
2. Creates initial placement groups and actors
3. Runs training steps while monitoring for new workers
4. Automatically scales up actors when new workers join

Usage:
    python training_loop_with_autoscaling.py
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import ray
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@dataclass
class ActorInfo:
    """Information about a training actor."""

    actor_ref: ray.actor.ActorHandle
    placement_group: PlacementGroup
    node_id: str
    actor_id: int


@ray.remote(num_gpus=1, num_cpus=1)
class TrainingActor:
    """A training actor that simulates model training on 1 GPU."""

    def __init__(self, actor_id: int):
        self.actor_id = actor_id
        self.training_steps = 0

        # Get node information
        import socket

        self.hostname = socket.gethostname()

        # Initialize GPU if available
        try:
            import torch

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.gpu_name = torch.cuda.get_device_name(0)
                print(
                    f"🤖 Training Actor {actor_id} initialized on {self.hostname} with GPU: {self.gpu_name}"
                )
            else:
                self.device = torch.device("cpu")
                print(
                    f"🤖 Training Actor {actor_id} initialized on {self.hostname} with CPU only"
                )
        except ImportError:
            self.device = "CPU only"
            print(
                f"🤖 Training Actor {actor_id} initialized on {self.hostname} (no PyTorch)"
            )

    def get_info(self) -> Dict:
        """Get information about this training actor."""
        return {
            "actor_id": self.actor_id,
            "hostname": self.hostname,
            "device": str(self.device),
            "training_steps": self.training_steps,
            "node_id": ray.get_runtime_context().get_node_id(),
        }

    def training_step(self, step_data: Dict) -> Dict:
        """Simulate a training step."""
        self.training_steps += 1

        start_time = time.time()

        try:
            import torch

            if str(self.device) == "cuda" and torch.cuda.is_available():
                # Simulate forward/backward pass
                device = torch.device("cuda")
                batch_size = step_data.get("batch_size", 32)

                # Simulate model forward pass
                x = torch.randn(batch_size, 512, device=device)
                weights = torch.randn(512, 256, device=device, requires_grad=True)
                y = torch.matmul(x, weights)

                # Simulate loss computation and backward pass
                loss = torch.mean(y**2)
                loss.backward()

                result = f"GPU training: loss={loss.item():.4f}"
            else:
                # CPU fallback
                result = (
                    f"CPU training: simulated_loss={0.5 + 0.1 * self.training_steps}"
                )
                time.sleep(1.0)  # Simulate training time

        except Exception as e:
            result = f"Training error: {e}"

        end_time = time.time()
        duration = end_time - start_time

        return {
            "actor_id": self.actor_id,
            "step": self.training_steps,
            "result": result,
            "duration": duration,
            "step_data": step_data,
        }


class DynamicTrainingManager:
    """Manages training actors with dynamic scaling."""

    def __init__(self):
        self.placement_groups: List[PlacementGroup] = []
        self.actors: List[ActorInfo] = []
        self.known_nodes: Dict[str, Dict] = {}
        self.next_actor_id = 0
        self.global_step = 0

    def connect_to_cluster(self):
        """Connect to existing Ray cluster."""
        try:
            ray.init(address="auto")
            print("🌟 Connected to existing Ray cluster")
        except Exception as e:
            print(f"❌ Failed to connect to Ray cluster: {e}")
            print("💡 Make sure to start the cluster first with ./start_ray_cluster.sh")
            return False

        # Print cluster info
        resources = ray.cluster_resources()
        print(f"🔧 Cluster resources: {resources}")
        return True

    def get_cluster_nodes(self) -> Dict[str, Dict]:
        """Get information about all nodes in the cluster."""
        nodes = {}

        for node in ray.nodes():
            if node.get("Alive", False):
                node_id = node["NodeID"]
                resources = node.get("Resources", {})

                nodes[node_id] = {
                    "node_id": node_id,
                    "resources": resources,
                    "alive": True,
                }

        return nodes

    def detect_new_workers(self) -> List[Dict]:
        """Detect newly joined workers since last check."""
        current_nodes = self.get_cluster_nodes()
        new_workers = []

        for node_id, node_info in current_nodes.items():
            if node_id not in self.known_nodes:
                # Check if this node has GPU (workers have GPU >= 1, head has 0)
                gpu_count = node_info["resources"].get("GPU", 0)

                if gpu_count >= 1:
                    new_workers.append(node_info)
                    print(
                        f"🆕 Detected new worker: {node_id[:10]}... with {gpu_count} GPUs"
                    )

        # Update known nodes
        self.known_nodes = current_nodes
        return new_workers

    def create_placement_group_for_node(self) -> Optional[PlacementGroup]:
        """Create a placement group for training."""
        try:
            pg = placement_group(
                bundles=[{"GPU": 1, "CPU": 1}],
                strategy="STRICT_PACK",
                name=f"training_pg_{len(self.placement_groups)}",
            )

            # Wait for placement group to be ready
            ready = ray.get(pg.ready(), timeout=30)
            if ready:
                self.placement_groups.append(pg)
                print(f"✅ Created placement group {len(self.placement_groups)}")
                return pg
            else:
                print("❌ Failed to create placement group")
                return None

        except Exception as e:
            print(f"❌ Error creating placement group: {e}")
            return None

    def spawn_training_actor(self, pg: PlacementGroup) -> Optional[ActorInfo]:
        """Spawn a training actor on the given placement group."""
        try:
            actor_id = self.next_actor_id
            self.next_actor_id += 1

            # Create actor with placement group scheduling
            actor = TrainingActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                )
            ).remote(actor_id)

            # Get actor info
            info = ray.get(actor.get_info.remote(), timeout=10)
            node_id = info["node_id"]

            actor_info = ActorInfo(
                actor_ref=actor, placement_group=pg, node_id=node_id, actor_id=actor_id
            )

            self.actors.append(actor_info)
            print(f"🎭 Spawned training actor {actor_id} on node {node_id[:10]}...")
            return actor_info

        except Exception as e:
            print(f"❌ Error spawning training actor: {e}")
            return None

    def initialize_training(self):
        """Initialize training with existing cluster nodes."""
        print("🔄 Initializing training...")

        # Detect initial nodes
        initial_nodes = self.get_cluster_nodes()
        print(f"📊 Found {len(initial_nodes)} initial nodes")

        # Create placement groups for nodes with GPUs (workers have GPU >= 1, head has 0)
        nodes_with_gpu = [
            node
            for node in initial_nodes.values()
            if node["resources"].get("GPU", 0) >= 1
        ]

        print(f"🎯 Creating {len(nodes_with_gpu)} training actors...")

        for i in range(len(nodes_with_gpu)):
            pg = self.create_placement_group_for_node()
            if pg:
                self.spawn_training_actor(pg)

        print(f"✅ Training initialized with {len(self.actors)} actors")

    def run_training_step(self):
        """Run a single training step on all actors."""
        if not self.actors:
            print("⚠️ No training actors available")
            return

        self.global_step += 1

        step_data = {
            "global_step": self.global_step,
            "batch_size": 32,
            "learning_rate": 0.001,
        }

        # Submit training steps to all actors
        print(
            f"🚀 Running training step {self.global_step} on {len(self.actors)} actors..."
        )

        futures = []
        for actor_info in self.actors:
            future = actor_info.actor_ref.training_step.remote(step_data)
            futures.append(future)

        # Wait for results
        try:
            results = ray.get(futures, timeout=30)

            # Print training results
            total_duration = sum(r["duration"] for r in results)
            avg_duration = total_duration / len(results)

            print(f"📊 Step {self.global_step} completed:")
            print(f"   - {len(results)} actors trained")
            print(f"   - Average duration: {avg_duration:.3f}s")

            for result in results:
                print(f"   - Actor {result['actor_id']}: {result['result']}")

        except Exception as e:
            print(f"⚠️ Training step failed: {e}")

    def check_and_scale(self):
        """Check for new workers and scale up if needed."""
        new_workers = self.detect_new_workers()

        for worker in new_workers:
            pg = self.create_placement_group_for_node()
            if pg:
                self.spawn_training_actor(pg)

    def run_training_loop(self, max_steps: int = 100):
        """Run the main training loop with dynamic scaling."""
        print(f"🏃 Starting training loop for {max_steps} steps...")
        print("💡 Training will check for new workers at the start of each step")
        print("🔄 To add workers: ./add_ray_worker.sh (in another terminal)")
        print()

        for step in range(max_steps):
            # Check for new workers at start of each step
            self.check_and_scale()

            # Run training step
            self.run_training_step()

            # Brief pause between steps
            time.sleep(2)

            if step % 10 == 9:
                print(f"📈 Completed {step + 1}/{max_steps} training steps")

        print("🎉 Training loop completed!")

    def shutdown(self):
        """Clean shutdown of all resources."""
        print("🧹 Cleaning up training resources...")

        # Kill actors
        for actor_info in self.actors:
            try:
                ray.kill(actor_info.actor_ref)
            except Exception:
                pass

        # Remove placement groups
        for pg in self.placement_groups:
            try:
                remove_placement_group(pg)
            except Exception:
                pass

        print("✅ Training cleanup complete")


def main():
    """Main function to run the training loop."""
    training_manager = DynamicTrainingManager()

    try:
        # Connect to Ray cluster
        if not training_manager.connect_to_cluster():
            return

        # Initialize training
        training_manager.initialize_training()

        print("\n" + "=" * 60)
        print("🎯 DYNAMIC TRAINING LOOP STARTED")
        print("=" * 60)
        print("💡 This training loop will:")
        print("   • Run training steps on all available actors")
        print("   • Monitor for new workers at each step")
        print("   • Automatically scale up when new workers join")
        print("   • Use 1 GPU per training actor")
        print("\n🔄 To add workers: ./add_ray_worker.sh")
        print("   Press Ctrl+C to stop training")
        print("=" * 60 + "\n")

        # Run training loop
        training_manager.run_training_loop(max_steps=50)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        training_manager.shutdown()


if __name__ == "__main__":
    main()
