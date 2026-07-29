# VAIR GitHub GPU localhost

This is the Apple GPU localhost application for
[`sahayu123/AIreferee`](https://github.com/sahayu123/AIreferee).

- UI: `http://localhost:3200`
- model API: `http://localhost:8200`
- source checkout: the repository root (`..`)
- source notebook: `AI Referee Foul Checker Prototype (1).ipynb`
- checkpoints: place the three trained `.pt` files in `models/`:
  `image_foul_mlp_v408.pt`, `resnet18_mixed.pt`, and `resnet_contact.pt`
- handball specialist: supplied Handball Detection Project trajectory,
  arm-collision, and arm-angle decision pipeline
- runtime modernization: YOLO11m ball detection and YOLOv8m-Pose skeletons
  run on Apple MPS in place of the archive's missing 2022 YOLOv5/HRNet weights
- automatic proximity call: a reliable ball-to-arm gap of at most 4% of the
  frame diagonal can award handball without the angle rule

The trained checkpoints are intentionally ignored by Git. To keep them
elsewhere, set `VAIR_CHECKPOINT_ROOT` to the directory containing all three
files. `VAIR_SOURCE_ROOT` can similarly override the notebook source directory.

Create the Python environment, then run the complete site:

```bash
npm run setup:models
npm run dev:full
```

The frontend runs at `http://localhost:3200`; the model API runs at
`http://localhost:8200`.

Pull the current branch before a future run:

```bash
npm run source:pull
```
