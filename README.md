# Virtual Wind Tunnel

Drop an STL into a live wind tunnel and watch the flow around it. The fluid solver runs on an NVIDIA GPU
inside a Docker container; any computer on your network is the viewer and control panel.

- 3D lattice-Boltzmann solver (D3Q19, regularized collision, Smagorinsky LES) written in [NVIDIA Warp](https://github.com/NVIDIA/warp)
- Streamline particles, slice planes (speed, pressure, vorticity, streamwise velocity), surface pressure on the model
- Live drag / lift coefficients (momentum exchange), moving ground belt for cars
- Grids from 0.5M up to 33M cells (as much as the GPU memory allows)

## Run on Unraid

Requires the **Nvidia Driver** plugin (Apps tab). Then either:

**Template** — copy `unraid/wind-tunnel.xml` to `/boot/config/plugins/dockerMan/templates-user/` on the
server, then Docker → Add Container → pick *wind-tunnel*. Set `NVIDIA_VISIBLE_DEVICES` to your GPU's UUID
(shown in the Nvidia Driver plugin page) or leave it as `all`.

**Manually** — Docker → Add Container:

| Field | Value |
|---|---|
| Repository | `ghcr.io/rickxxrolling/wind-tunnel:latest` |
| Extra Parameters | `--runtime=nvidia` |
| Port | container `8765` → host `8765` |
| Path | container `/cache` → `/mnt/user/appdata/wind-tunnel` |
| Variable | `NVIDIA_VISIBLE_DEVICES` = `all` (or the GPU UUID) |
| Variable | `NVIDIA_DRIVER_CAPABILITIES` = `compute,utility` |

**Compose** — `docker compose up -d` with the included `docker-compose.yml`.

Open `http://<unraid-ip>:8765` from any browser. The header shows the GPU the solver found. The first start
compiles the CUDA kernels (cached in `/cache` afterwards).

## Run locally (development)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python server/app.py
```

Without CUDA (e.g. on a Mac) Warp falls back to the CPU and only the small grids are offered.

## Layout

- `server/lbm.py` — Warp kernels: generated, fully unrolled stream/collide, forces, particles, slices, surface pressure
- `server/app.py` — aiohttp server: solver thread, WebSocket streaming (quantized particles, slice images, forces)
- `web/` — Three.js viewer: STL loading, orientation, voxelization (sent to the server), rendering, UI

STLs are assumed Z-up and in millimetres (used for the default model length / Reynolds number); use the
X/Y/Z/Flip buttons if a model comes in sideways. Results are qualitative: the grid is coarse relative to
real CFD and viscosity is floored for stability, so Cd/Cl are ballpark values.
