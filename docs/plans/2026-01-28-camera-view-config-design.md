# Camera View Configuration Design

## Goal
Provide a CLI-driven, user-friendly way to map head/wrist camera views to device IDs, and propagate that mapping to the WebXR client so panels and video tracks line up deterministically.

## Architecture
- CLI flags build a `camera_views` config in the Python launcher.
- `Teleop` sends `camera_views` in the existing WebSocket `config` message and uses it to build `video_config` stream IDs.
- WebXR reads `camera_views` from the config message to set panel visibility and to route incoming video tracks by `trackId`.

## Components & API
- CLI flags (explicit mapping only):
  - `--head-device <idx|/dev/...>`
  - `--wrist-left-device <idx|/dev/...>`
  - `--wrist-right-device <idx|/dev/...>`
- Parsing normalizes to `{ view_key: { device } }` with view keys: `head`, `wrist_left`, `wrist_right`.
- `Teleop` owns:
  - `camera_views` config (for WS `config` message)
  - `video_config.streams` IDs matching view keys (for track routing)
- WebXR:
  - stores `camera_views` on receipt
  - hides any panel with no mapped device
  - routes tracks by `trackId` to the matching panel (order fallback only if `trackId` missing)

## Data Flow
1. User runs `uv run python -m teleop.basic --head-device 0 --wrist-left-device /dev/video1`.
2. CLI builds `camera_views` and passes it into `Teleop`.
3. On WS connect, server sends:
   - `config`: `{ input_mode, camera_views }`
   - `video_config`: `{ streams: [{ id: "head", device: 0 }, { id: "wrist_left", device: "/dev/video1" }] }`
4. WebXR receives `config`, applies visibility and stores mapping.
5. Video tracks arrive with `trackId` matching view keys, and the client assigns them to panels.

## Error Handling
- Invalid device specs (non-int, non-`/dev/*`) cause a clear CLI error and exit.
- Duplicate device mapping is allowed but logged as a warning.
- If a device fails to open, the server logs a warning tagged with the view key and continues other streams.
- If a track never arrives, the panel remains in placeholder state without crashing.

## Testing
- Python: pytest unit tests for device parsing and `camera_views` assembly.
- WebXR: minimal unit tests to verify panel visibility and track routing by `trackId`.
- If no TS test runner exists, add a minimal one scoped to these new helpers only.
