# Third-party notices

`tpsreg` itself is licensed under the MIT License (see [LICENSE](LICENSE)).

It vendors third-party source code under `src/tpsreg/Matchanything/`, used to
provide optional automatic control point detection. That code keeps its own
licenses, summarised below. All of it is permissively licensed (Apache-2.0 or
MIT), so the repository and the built artifacts can be redistributed freely.

The vendored tree has been reduced to the inference path only. Upstream ships
training scripts, benchmarks, dataset builders, COLMAP helpers, demos and
notebooks; none of it is reachable from the model this project loads, so it was
removed. What remains is the 64 modules needed to construct the RoMa matcher and
run it on an image pair, plus the upstream `LICENSE` and `README.md` files.

## Component licenses

| Component | Path | License |
|---|---|---|
| [MatchAnything](https://github.com/zju3dv/MatchAnything) | `src/tpsreg/Matchanything/` | Apache License 2.0 — see `src/tpsreg/Matchanything/LICENSE` |
| [RoMa](https://github.com/Parskatt/RoMa) | `src/tpsreg/Matchanything/third_party/ROMA/` | MIT, © 2023 Johan Edstedt — see `src/tpsreg/Matchanything/third_party/ROMA/LICENSE` |
| [DINOv2](https://github.com/facebookresearch/dinov2) | `.../third_party/ROMA/roma/models/transformer/` | Apache License 2.0, © Meta Platforms, Inc. |

## Removed: non-commercial components

Upstream RoMa also ships [CroCo](https://github.com/naver/croco) and
[DUSt3R](https://github.com/naver/dust3r) under
`roma/models/croco/` and `roma/models/dust3r/`. Both are licensed
**CC BY-NC-SA 4.0 (non-commercial use only)** by Naver Corporation, which would
have prevented this project from being redistributed under its MIT license.

Neither is reachable from the code path this project uses. The RoMa model
loaded by `tpsreg.roma_matcher` resolves through `roma.models.matcher`,
`roma.models.encoders` and `roma.models.transformer`, none of which import
CroCo or DUSt3R. Those two subtrees were therefore removed rather than shipped.

If you re-vendor RoMa from upstream, drop `roma/models/croco/` and
`roma/models/dust3r/` again, or the non-commercial restriction returns. The
release workflow fails the build if any `CC BY-NC-SA` file reappears under
`src/`.

## Re-vendoring

If you need to update the vendored model, the import graph is what decides
what to keep. Walk it from the four entry points this project actually loads —
`src/lightning/lightning_loftr.py`, `src/config/default.py`,
`configs/models/roma_model.py` and `Matchanything/__init__.py` — following
absolute, relative and star imports, and including every ancestor
`__init__.py` along the way. Anything the walk does not reach is not used at
inference time.

Note that upstream imports are rewritten from `Matchanything.` to
`tpsreg.Matchanything.` so the tree resolves as an installed package rather
than relying on the working directory.

## Theme

The Tk theme under `src/tpsreg/resources/theme/` is the
[Azure theme](https://github.com/rdbende/Azure-ttk-theme) by rdbende, MIT
licensed.

## Model weights

The MatchAnything checkpoint is **not** distributed with this package. It is
downloaded separately by the user and is subject to its own upstream terms.

## Citing MatchAnything

If you use the automatic detection feature in published work, please cite the
upstream authors as well as this tool:

```bibtex
@article{he2025matchanything,
  title   = {MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training},
  author  = {He, Xingyi and Yu, Hao and Peng, Sida and Tan, Dongli and Shen, Zehong and Bao, Hujun and Zhou, Xiaowei},
  journal = {arXiv preprint arXiv:2501.07556},
  year    = {2025}
}
```
