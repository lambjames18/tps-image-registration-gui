# Third-party notices

`tpsreg` itself is licensed under the MIT License (see [LICENSE](LICENSE)).

It vendors third-party source code under `src/tpsreg/Matchanything/`, used to
provide optional automatic control point detection. That code is **not** covered
by the MIT license above. Its licenses are summarised below.

## ⚠️ Non-commercial code is included

Part of the vendored tree is licensed **CC BY-NC-SA 4.0, which permits
non-commercial use only**. 47 files under
`src/tpsreg/Matchanything/third_party/ROMA/roma/models/` carry this restriction,
specifically the `croco/` and `dust3r/` subtrees:

> Copyright (C) 2022-present Naver Corporation. All rights reserved.
> Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

Consequences worth understanding before redistributing:

- The repository as a whole **cannot** be treated as MIT-licensed, even though
  `tpsreg`'s own code is. Anyone redistributing the combined work inherits the
  non-commercial restriction on those files.
- Publishing the current wheel to a public index distributes those files, since
  the whole `Matchanything` tree is packaged.
- Academic and other non-commercial research use is permitted. Commercial use of
  the vendored model code is not.

If unrestricted redistribution matters for your use, the options are to remove
the vendored tree and load MatchAnything from a separately installed upstream
package, or to package it as an optional plugin distributed separately from the
MIT-licensed core. `tpsreg`'s own registration code has no such restriction, and
the core install does not depend on any of it.

## Component licenses

| Component | Path | License |
|---|---|---|
| [MatchAnything](https://github.com/zju3dv/MatchAnything) | `src/tpsreg/Matchanything/` | Apache License 2.0 — see `src/tpsreg/Matchanything/LICENSE` |
| [RoMa](https://github.com/Parskatt/RoMa) | `src/tpsreg/Matchanything/third_party/ROMA/` | MIT, © 2023 Johan Edstedt — see `src/tpsreg/Matchanything/third_party/ROMA/LICENSE` |
| [CroCo](https://github.com/naver/croco) | `.../third_party/ROMA/roma/models/croco/` | **CC BY-NC-SA 4.0**, © Naver Corporation — non-commercial use only |
| [DUSt3R](https://github.com/naver/dust3r) | `.../third_party/ROMA/roma/models/dust3r/` | **CC BY-NC-SA 4.0**, © Naver Corporation — non-commercial use only |
| [DINOv2](https://github.com/facebookresearch/dinov2) | `.../third_party/ROMA/roma/models/transformer/` | Apache License 2.0, © Meta Platforms, Inc. |

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
