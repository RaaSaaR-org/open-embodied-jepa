# Draft upstream report: intermittent hang in `UsdPhysics.LoadUsdPhysicsFromRange` during Newton `add_usd` (Isaac Lab 3 / Isaac Sim 6.0.0-rc.22)

**Status: internal record, not posted.** A version edited for upstream, without project paths, is prepared separately for the owner to post. It was written for the Newton and Isaac Lab issue trackers. The project owner posts it after checking it. Evidence and context are in [ISAAC_E9_REPLAY.md](../ISAAC_E9_REPLAY.md) §4 and [ISAAC_NEWTON_SPIKE.md](../ISAAC_NEWTON_SPIKE.md) §6. Before posting, remove the project-internal paths, or replace them with a public minimal repro if one is made.

---

## Summary

Building a Newton model from the live Isaac Lab stage with `newton.ModelBuilder.add_usd(stage, ...)` sometimes never returns. It hangs in about 1 of 5 process starts: 4 of 22 in our runs. Every stack dump taken during a hang shows the main thread inside `UsdPhysics.LoadUsdPhysicsFromRange`, called from `newton/_src/utils/import_usd.py:317` (`parse_usd`). The inputs were the same on every start: the same USD, code and container image. A start that gets past this point never hung later.

## Update 2026-09-30: `PXR_WORK_THREAD_LIMIT=1` (0 hangs in 24 starts)

Newton already documents an OpenUSD thread-safety bug in this function ([newton#1743](https://github.com/newton-physics/newton/issues/1743), [#2216](https://github.com/newton-physics/newton/issues/2216); `docs/concepts/usd_parsing.rst`, "Limitations"): a crash when many mesh colliders sit under one rigid body, fixed by OpenUSD commit `ed857d77` (PR 4002, in v26.05-rc1), with `PXR_WORK_THREAD_LIMIT=1` as the workaround. We tested that workaround. Details: [ISAAC_E9_REPLAY.md](../ISAAC_E9_REPLAY.md) §4, "The OpenUSD thread limit".

- **USD in the running Kit process:** 0.25.11 (`omni.usd.libs-1.0.1`). Stock v25.11 does not contain `ed857d77`; `UsdPhysics.LoadStageFromPrimRange` (the 26.05 name) is absent. A vendor backport was not checked.
- **The env var alone has no effect in Isaac Sim.** Kit sets `PXR_WORK_THREAD_LIMIT` itself before USD reads it (`SimulationApp`: min(cores, `limit_cpu_threads`); `omni.usd.config`: `"16"`), so `docker run -e PXR_WORK_THREAD_LIMIT=1` still gave `Work.GetConcurrencyLimit() == 16`, and `Work.SetConcurrencyLimit(1)` cannot override the variable afterwards. Our 22 earlier starts therefore ran with 16 work threads. We pin the variable in `os.environ` before the app starts and log `Work.GetConcurrencyLimit()`: it was 1 in every pinned start.
- **Result:** 0 hangs in 24 start-up-only starts with the limit at 1, against 4 in 22 before (one-sided Fisher exact p = 0.045; with the pin, a hang rate up to 12 % is not excluded, one-sided 95 % bound). Suggestive, not proof; the earlier starts are a historical comparison, not a concurrent arm.
- **Cost:** start-up 174 s against 126 s (median, 24 pinned against 4 unpinned starts on the same day); `add_usd` 11.6 s against 9.4 s; one simulation step 32.0 against 27.2 ms (median, 2 attempts each).
- **Reading:** this looks like a hang or livelock form of the same OpenUSD race (a spinning main thread, not a crash), on a live Kit stage (*inferred*; no native stack).

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

## Earlier hypothesis (before the update above)

`add_usd` parses the **live** Kit stage. Kit's background threads may modify or iterate the same stage at the same time, for example Fabric/USDRT sync or the physics stage-update listener. That would be a race. One way to test it is to parse a detached copy (`stage.Flatten()` into an anonymous in-memory stage) and compare hang rates. At about 1 hang in 5 starts, that needs roughly 20 or more starts per arm to be informative. We have not run it.

## Workaround we use

- `PXR_WORK_THREAD_LIMIT=1`, pinned against Kit's own setting (update above). It costs about 48 s of start-up and about 18 % step time.
- A start-up watchdog stays as a backstop: if model building has not finished within 480 s, it stops that container and starts a new one, up to a bounded number of tries. This is safe for us because the hang happens before any episode runs.

## Questions for maintainers

1. Is this hang the same OpenUSD race as #1743 (which is documented as a crash)? If so, should the "Limitations" section of `usd_parsing.rst` also mention hangs, and that inside Isaac Sim / Kit the environment variable is overwritten at start-up, so it must be pinned (or Kit's setting changed) to take effect?
2. Is `ModelBuilder.add_usd` supported on the live stage while the Kit app is running? Or should it be given a flattened or detached stage?
3. Would a native stack from a build with symbols help? We can collect one if we can get ptrace access inside the container.
