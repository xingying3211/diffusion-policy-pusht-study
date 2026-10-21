# Upstream attribution

This is an evaluation and analysis study built on existing work.

| Component | Reference | License/source of evidence |
| --- | --- | --- |
| LeRobot 0.3.3 | [versioned repository](https://github.com/huggingface/lerobot/tree/v0.3.3) | Apache-2.0; installed distribution license retained in `licenses/lerobot-LICENSE.txt` |
| gym-pusht 0.1.5 | [repository](https://github.com/huggingface/gym-pusht) | Apache-2.0; installed distribution license retained in `licenses/gym-pusht-LICENSE.txt` |
| Pretrained `lerobot/diffusion_pusht` | [model](https://huggingface.co/lerobot/diffusion_pusht) | Original model metadata declares Apache-2.0; weights are not redistributed |
| Diffusion Policy | [project and authors](https://diffusion-policy.cs.columbia.edu/) | Algorithm credit belongs to the original authors; no original algorithm source is vendored |

The network, normalization, diffusion scheduler, evaluator, environment physics,
and pretrained weights are upstream work. The project-local wrappers call and
instrument LeRobot's existing APIs; they do not implement a new learning method.
Third-party dependency licenses continue to apply to their respective packages.

No new blanket license has been selected for the project-local materials during
curation. Public visibility alone would not grant an open-source reuse license
for those materials. The included dependency licenses do not relicense this
entire repository.
