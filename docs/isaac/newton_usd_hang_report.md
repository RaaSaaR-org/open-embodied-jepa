# Draft upstream report: intermittent hang in `UsdPhysics.LoadUsdPhysicsFromRange` during Newton `add_usd` (Isaac Lab 3 / Isaac Sim 6.0.0-rc.22)

**Status: internal record, not posted.** A version edited for upstream, without project paths, is prepared separately for the owner to post. It was written for the Newton and Isaac Lab issue trackers. The project owner posts it after checking it. Evidence and context are in [ISAAC_E9_REPLAY.md](../ISAAC_E9_REPLAY.md) §4 and [ISAAC_NEWTON_SPIKE.md](../ISAAC_NEWTON_SPIKE.md) §6. Before posting, remove the project-internal paths, or replace them with a public minimal repro if one is made.

---

## Summary

Building a Newton model from the live Isaac Lab stage with `newton.ModelBuilder.add_usd(stage, ...)` sometimes never returns. It hangs in about 1 of 5 process starts: 4 of 22 in our runs. Every stack dump taken during a hang shows the main thread inside `UsdPhysics.LoadUsdPhysicsFromRange`, called from `newton/_src/utils/import_usd.py:317` (`parse_usd`). The inputs were the same on every start: the same USD, code and container image. A start that gets past this point never hung later.

## Environment

- Container image: `isaaclab_arena:latest`, `sha256:2588b52605d77552d4480196501c4b8b61774ef6d70d9b3d59291622d6856d9c`. It was built locally, apparently from the `isaac-sim/IsaacLab-Arena` checkout on this host: commit `8b4a3a47…`, `docker/Dockerfile.isaaclab_arena`, `INSTALL_GROOT=false`, IsaacLab submodule `e57379c6…`. The image was created on 2026-09-23. That the image came from this exact checkout is inferred from the image layers and has not been verified.
- Isaac Sim `6.0.0-rc.22+release.33481.407f3ea1.gl` (`/isaac-sim/VERSION`).
- Isaac Lab 3.0.0 and `isaaclab_newton` 0.5.9.
- `newton` 1.1.0.dev0, `mujoco_warp` 3.5.0.2, `mujoco` 3.5.0 and `warp` 1.12.0. These are the image's installed packages under `/isaac-sim/kit/python/lib/python3.12/site-packages/`.
- Python 3.12, from Isaac Sim's bundled interpreter (`/isaac-sim/python.sh`).
- Host: Ubuntu 24.04.3 LTS, kernel 7.0.0-34-generic, NVIDIA driver 595.91.07, one RTX 5080 (16 GB). Another process on the same GPU held about 6.6 GB during the runs.
- Docker run flags: `--gpus all --runtime=nvidia --ipc=host --net=host`. The app runs headless (`AppLauncher`, `headless=True`) with rendering off.

## What we run

These steps run in one process, inside the container:

1. Start the app with `isaaclab.app.AppLauncher(args).app`, headless.
2. Build a scene on the Kit stage with Isaac Lab:
   - a Unitree G1 29-DoF robot with two Dex3 hands, from a USD converted from MJCF with Isaac Lab's MJCF converter (`allow_self_collision: False`);
   - a table, a plate and a sphere, as rigid objects.
   Isaac Lab's physics manager is set to Newton.
3. Before `sim.reset()`, build the Newton model ourselves and hand it to `NewtonManager.set_builder`. We do this to register the MuJoCo custom attributes, which Isaac Lab's default builder does not register:

   ```python
   from newton import ModelBuilder
   from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx
   from newton.solvers import SolverMuJoCo

   stage = isaaclab.sim.utils.get_current_stage()
   builder = ModelBuilder(up_axis="Z")
   SolverMuJoCo.register_custom_attributes(builder)
   builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])  # hangs here
   ```

4. Then `NewtonManager.set_builder(builder)`, `sim.reset()`, and the solver's Warp kernel compilation and CUDA graph capture follow.

Repro steps with our code: start the physics server `e9_server_isaac.py --startup_only --physics newton` in a fresh container, and repeat. A healthy start logs `transport built` after 125–152 s. A hung start never does.

We have **not** made a minimal standalone repro (just the robot USD, without our scene objects or our code). We do not yet know whether one hangs at the same rate.

## Frequency

| Series | Newton starts | Hung |
| --- | --- | --- |
| Earlier spike (full runs) | 10 | 2 |
| This series: 3 full server starts, 8 start-up-only probes and 1 start-up-only start through our watchdog, one container at a time | 12 | 2 |
| Total | 22 | 4 |

- Every start used the same USD, code, image and command.
- Healthy starts reached `transport built` in 124.9–151.6 s (the 10 healthy starts of this series). Most of that time is Warp kernel compilation and CUDA graph capture (`CUDA graph took: 83 s`) into an empty per-container kernel cache.
- Hung starts were stopped after 7 to 20+ minutes with nothing further logged.
- After a successful build we saw no hang: two servers ran 50 752 control steps each.
- We did not test whether PhysX starts can hang in the same way.

## Stack excerpts

`faulthandler.dump_traceback_later(120, repeat=True)` was armed during start-up. All 8 dumps from one hung server, over 16 minutes, and all 3 dumps from one hung probe, over 7 minutes, show the same main-thread stack. Our frames are at the end.

```
Thread 0x00007ed55c7f8780 (most recent call first):
  File ".../site-packages/newton/_src/utils/import_usd.py", line 317 in parse_usd
  File ".../site-packages/newton/_src/sim/builder.py", line 2403 in add_usd
  File "/oej/src/embodied_jepa/isaac_transport.py", line 657 in _install_newton_builder
  File "/oej/src/embodied_jepa/isaac_transport.py", line 327 in __init__
  File "/oej/scripts/isaac/e9_server_isaac.py", line 110 in main
```

`import_usd.py` line 317 is:

```python
ret_dict = UsdPhysics.LoadUsdPhysicsFromRange(stage, [root_path], excludePaths=...)
```

The two hangs in the earlier spike had no stack dump. Their logs stop at the same position: after Warp initialised and before the model was finalised.

## Process state during a hang

- The main thread, which is the process's own PID, used one full core in user space. It had no wait channel (`/proc/<pid>/task/<tid>/stat`, `wchan`).
- No other thread was busy.
- RSS stayed flat at 2.33 GB.
- This looks like a livelock or an infinite loop inside the C++ `LoadUsdPhysicsFromRange` parse. It does not look like a lock wait (a sleeping thread) or a crash.
- We could not take a native stack: `ptrace_scope` is 1 and the process is not a child of our shell, so py-spy and gdb could not attach.

## Hypothesis (untested)

`add_usd` parses the **live** Kit stage. Kit's background threads may modify or iterate the same stage at the same time, for example Fabric/USDRT sync or the physics stage-update listener. That would be a race. One way to test it is to parse a detached copy (`stage.Flatten()` into an anonymous in-memory stage) and compare hang rates. At about 1 hang in 5 starts, that needs roughly 20 or more starts per arm to be informative. We have not run it.

## Workaround we use

A start-up watchdog: if `transport built` is not logged within 300 s, the watchdog stops only that container and starts a new one, up to a bounded number of tries. This is safe for us because the hang happens before any episode runs.

## Questions for maintainers

1. Is `ModelBuilder.add_usd` supported on the live stage while the Kit app is running? Or should it be given a flattened or detached stage?
2. Is there a known issue with `UsdPhysics.LoadUsdPhysicsFromRange` looping on some stage states, for example an articulation with many `excludePaths`, or while a stage is being edited?
3. Would a native stack from a build with symbols help? We can collect one if we can get ptrace access inside the container.
