# {py:mod}`nemo_rl.distributed.worker_groups`

```{py:module} nemo_rl.distributed.worker_groups
```

```{autodoc2-docstring} nemo_rl.distributed.worker_groups
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiWorkerFuture <nemo_rl.distributed.worker_groups.MultiWorkerFuture>`
  - ```{autodoc2-docstring} nemo_rl.distributed.worker_groups.MultiWorkerFuture
    :summary:
    ```
* - {py:obj}`RayWorkerBuilder <nemo_rl.distributed.worker_groups.RayWorkerBuilder>`
  - ```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerBuilder
    :summary:
    ```
* - {py:obj}`RayWorkerGroup <nemo_rl.distributed.worker_groups.RayWorkerGroup>`
  - ```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup
    :summary:
    ```
````

### API

`````{py:class} MultiWorkerFuture
:canonical: nemo_rl.distributed.worker_groups.MultiWorkerFuture

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.MultiWorkerFuture
```

````{py:attribute} futures
:canonical: nemo_rl.distributed.worker_groups.MultiWorkerFuture.futures
:type: list[ray.ObjectRef]
:value: >
   None

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.MultiWorkerFuture.futures
```

````

````{py:attribute} return_from_workers
:canonical: nemo_rl.distributed.worker_groups.MultiWorkerFuture.return_from_workers
:type: typing.Optional[list[int]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.MultiWorkerFuture.return_from_workers
```

````

````{py:attribute} called_workers
:canonical: nemo_rl.distributed.worker_groups.MultiWorkerFuture.called_workers
:type: typing.Optional[list[int]]
:value: >
   None

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.MultiWorkerFuture.called_workers
```

````

````{py:method} get_results(worker_group: nemo_rl.distributed.worker_groups.RayWorkerGroup, return_generators_as_proxies: bool = False) -> list[typing.Any]
:canonical: nemo_rl.distributed.worker_groups.MultiWorkerFuture.get_results

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.MultiWorkerFuture.get_results
```

````

`````

``````{py:class} RayWorkerBuilder(ray_actor_class_fqn: str, *args, **kwargs)
:canonical: nemo_rl.distributed.worker_groups.RayWorkerBuilder

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerBuilder
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerBuilder.__init__
```

`````{py:class} IsolatedWorkerInitializer(ray_actor_class_fqn: str, *init_args, **init_kwargs)
:canonical: nemo_rl.distributed.worker_groups.RayWorkerBuilder.IsolatedWorkerInitializer

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerBuilder.IsolatedWorkerInitializer
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerBuilder.IsolatedWorkerInitializer.__init__
```

````{py:method} create_worker(placement_group: ray.util.placement_group.PlacementGroup, placement_group_bundle_index: int, num_gpus: int, bundle_indices: typing.Optional[tuple] = None, **extra_options: typing.Optional[dict[str, typing.Any]])
:canonical: nemo_rl.distributed.worker_groups.RayWorkerBuilder.IsolatedWorkerInitializer.create_worker

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerBuilder.IsolatedWorkerInitializer.create_worker
```

````

`````

````{py:method} create_worker_async(placement_group: ray.util.placement_group.PlacementGroup, placement_group_bundle_index: int, num_gpus: float | int, bundle_indices: typing.Optional[tuple[int, list[int]]] = None, **extra_options: typing.Any) -> tuple[ray.ObjectRef, ray.actor.ActorHandle]
:canonical: nemo_rl.distributed.worker_groups.RayWorkerBuilder.create_worker_async

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerBuilder.create_worker_async
```

````

````{py:method} __call__(placement_group: ray.util.placement_group.PlacementGroup, placement_group_bundle_index: int, num_gpus: float | int, bundle_indices: typing.Optional[tuple[int, list[int]]] = None, **extra_options: typing.Any) -> ray.actor.ActorHandle
:canonical: nemo_rl.distributed.worker_groups.RayWorkerBuilder.__call__

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerBuilder.__call__
```

````

``````

`````{py:class} RayWorkerGroup(cluster: nemo_rl.distributed.virtual_cluster.RayVirtualCluster, remote_worker_builder: nemo_rl.distributed.worker_groups.RayWorkerBuilder, workers_per_node: typing.Optional[typing.Union[int, list[int]]] = None, name_prefix: str = '', bundle_indices_list: typing.Optional[list[tuple[int, list[int]]]] = None, sharding_annotations: typing.Optional[nemo_rl.distributed.named_sharding.NamedSharding] = None)
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.__init__
```

````{py:method} get_dp_leader_worker_idx(dp_shard_idx: int) -> int
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.get_dp_leader_worker_idx

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.get_dp_leader_worker_idx
```

````

````{py:method} _create_workers_from_bundle_indices(remote_worker_builder: nemo_rl.distributed.worker_groups.RayWorkerBuilder, bundle_indices_list: list[tuple[int, list[int]]]) -> None
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup._create_workers_from_bundle_indices

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup._create_workers_from_bundle_indices
```

````

````{py:property} workers
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.workers
:type: list[ray.actor.ActorHandle]

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.workers
```

````

````{py:property} worker_metadata
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.worker_metadata
:type: list[dict[str, typing.Any]]

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.worker_metadata
```

````

````{py:property} dp_size
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.dp_size
:type: int

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.dp_size
```

````

````{py:method} run_single_worker_single_data(method_name: str, worker_idx: int, *args, **kwargs) -> ray.ObjectRef
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.run_single_worker_single_data

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.run_single_worker_single_data
```

````

````{py:method} run_all_workers_multiple_data(method_name: str, *args, run_rank_0_only_axes: list[str] | None = None, common_kwargs: typing.Optional[dict[str, typing.Any]] = None, **kwargs) -> list[ray.ObjectRef]
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.run_all_workers_multiple_data

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.run_all_workers_multiple_data
```

````

````{py:method} run_all_workers_single_data(method_name: str, *args, run_rank_0_only_axes: list[str] | None = None, **kwargs) -> list[ray.ObjectRef]
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.run_all_workers_single_data

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.run_all_workers_single_data
```

````

````{py:method} run_all_workers_sharded_data(method_name: str, *args, in_sharded_axes: list[str] | None = None, replicate_on_axes: list[str] | None = None, output_is_replicated: list[str] | None = None, make_dummy_calls_to_free_axes: bool = False, common_kwargs: typing.Optional[dict[str, typing.Any]] = None, **kwargs) -> nemo_rl.distributed.worker_groups.MultiWorkerFuture
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.run_all_workers_sharded_data

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.run_all_workers_sharded_data
```

````

````{py:method} get_all_worker_results(future_bundle: nemo_rl.distributed.worker_groups.MultiWorkerFuture, return_generators_as_proxies: bool = False) -> list[typing.Any]
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.get_all_worker_results

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.get_all_worker_results
```

````

````{py:method} shutdown(cleanup_method: typing.Optional[str] = None, timeout: typing.Optional[float] = 30.0, force: bool = False) -> bool
:canonical: nemo_rl.distributed.worker_groups.RayWorkerGroup.shutdown

```{autodoc2-docstring} nemo_rl.distributed.worker_groups.RayWorkerGroup.shutdown
```

````

`````
