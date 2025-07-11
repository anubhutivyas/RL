# {py:mod}`nemo_rl.utils.nsys`

```{py:module} nemo_rl.utils.nsys
```

```{autodoc2-docstring} nemo_rl.utils.nsys
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProfilablePolicy <nemo_rl.utils.nsys.ProfilablePolicy>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`maybe_gpu_profile_step <nemo_rl.utils.nsys.maybe_gpu_profile_step>`
  - ```{autodoc2-docstring} nemo_rl.utils.nsys.maybe_gpu_profile_step
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NRL_NSYS_WORKER_PATTERNS <nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS>`
  - ```{autodoc2-docstring} nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS
    :summary:
    ```
* - {py:obj}`NRL_NSYS_PROFILE_STEP_RANGE <nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE>`
  - ```{autodoc2-docstring} nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE
    :summary:
    ```
````

### API

````{py:data} NRL_NSYS_WORKER_PATTERNS
:canonical: nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS
:value: >
   'get(...)'

```{autodoc2-docstring} nemo_rl.utils.nsys.NRL_NSYS_WORKER_PATTERNS
```

````

````{py:data} NRL_NSYS_PROFILE_STEP_RANGE
:canonical: nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE
:value: >
   'get(...)'

```{autodoc2-docstring} nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE
```

````

`````{py:class} ProfilablePolicy
:canonical: nemo_rl.utils.nsys.ProfilablePolicy

Bases: {py:obj}`typing.Protocol`

````{py:method} start_gpu_profiling() -> None
:canonical: nemo_rl.utils.nsys.ProfilablePolicy.start_gpu_profiling

```{autodoc2-docstring} nemo_rl.utils.nsys.ProfilablePolicy.start_gpu_profiling
```

````

````{py:method} stop_gpu_profiling() -> None
:canonical: nemo_rl.utils.nsys.ProfilablePolicy.stop_gpu_profiling

```{autodoc2-docstring} nemo_rl.utils.nsys.ProfilablePolicy.stop_gpu_profiling
```

````

`````

````{py:function} maybe_gpu_profile_step(policy: nemo_rl.utils.nsys.ProfilablePolicy, step: int)
:canonical: nemo_rl.utils.nsys.maybe_gpu_profile_step

```{autodoc2-docstring} nemo_rl.utils.nsys.maybe_gpu_profile_step
```
````
