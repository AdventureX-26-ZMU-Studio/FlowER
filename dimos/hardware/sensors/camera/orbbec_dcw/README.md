# Orbbec DaBai DCW

This integration contains two camera owners. Run only one at a time:

- `OrbbecDCWCamera`: a DimOS module publishing color, depth, camera info, TF,
  and optional point clouds.
- `dimos-dcw-vision`: a standalone MediaPipe web service on port `8890`.

The GB10 is `linux_aarch64`, so install the matching Python 3.12
`pyorbbecsdk` wheel supplied with the Orbbec SDK and make its native libraries
visible through `LD_LIBRARY_PATH`.

The standalone visual service additionally needs a MediaPipe build that
supports Python 3.12 on `linux_aarch64`.

```bash
export LD_LIBRARY_PATH=/home/asus/orbbec_sdk_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
dimos-dcw-vision --host 0.0.0.0 --port 8890
```

Endpoints:

- `/` web dashboard
- `/s` or `/status` JSON status
- `/s/color` annotated color MJPEG
- `/s/depth` depth MJPEG
- `/healthz` service health
- `/tracking` localhost-only POST endpoint used by the Viser demo panel

## Face-follow demo

The optional face-follow controller is disabled by default. The Viser panel
enables it through the localhost-only `/tracking` endpoint. It publishes
bounded angular and forward/back commands to `/coordinator_ee_twist_command` for the
`eef_twist_arm` control task.

Safety behavior:

- requires three consecutive face detections before moving
- samples aligned depth around the detected face and maintains a `1.0 m` target
- limits forward/back motion to `0.04 m/s`, yaw to `0.12 rad/s`, and pitch to
  `0.10 rad/s`
- smooths image/depth measurements and rate-limits command acceleration
- immediately zeros forward/back motion when aligned face depth is missing,
  closer than `0.35 m`, or farther than `3.0 m`
- sends zero velocity immediately when the face is lost or tracking is disabled
- the coordinator task also stops on command timeout
- automatically disables after 60 seconds and must be re-enabled by the operator

The DCW is frequently connected through USB 2.0. Defaults therefore use
`640x360 @ 15 FPS`; increase the rate only after verifying stable depth frames.
The MJPEG color/depth preview and face-follow control share the same capture
pipeline and can be used at the same time.
