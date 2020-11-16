
### Ablation studies on QuerYD

We conduct several ablation studies to investigate the importance of different components in the Collaborative Experts design.  Each ablation is conducted on the QuerYD dataset. 


| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| Scene    | t2v  | {{queryd-train-full-ce-only-scene.t2v}} | {{queryd-train-full-ce-only-scene.params}} | [config]({{queryd-train-full-ce-only-scene.config}}), [model]({{queryd-train-full-ce-only-scene.model}}), [log]({{queryd-train-full-ce-only-scene.log}}) |
| Scene + Inst. | t2v  | {{queryd-train-full-ce-only-scene-inst.t2v}} | {{queryd-train-full-ce-only-scene-inst.params}} | [config]({{queryd-train-full-ce-only-scene-inst.config}}), [model]({{queryd-train-full-ce-only-scene-inst.model}}), [log]({{queryd-train-full-ce-only-scene-inst.log}}) |
| Scene + r2p1d | t2v  | {{queryd-train-full-ce-only-scene-r2p1d.t2v}} | {{queryd-train-full-ce-only-scene-r2p1d.params}} | [config]({{queryd-train-full-ce-only-scene-r2p1d.config}}), [model]({{queryd-train-full-ce-only-scene-r2p1d.model}}), [log]({{queryd-train-full-ce-only-scene-r2p1d.log}}) |
| Scene + Audio | t2v  | {{queryd-train-full-ce-only-scene-audio.t2v}} | {{queryd-train-full-ce-only-scene-audio.params}} | [config]({{queryd-train-full-ce-only-scene-audio.config}}), [model]({{queryd-train-full-ce-only-scene-audio.model}}), [log]({{queryd-train-full-ce-only-scene-audio.log}}) |
| Scene    | v2t  | {{queryd-train-full-ce-only-scene.v2t}} | {{queryd-train-full-ce-only-scene.params}} | [config]({{queryd-train-full-ce-only-scene.config}}), [model]({{queryd-train-full-ce-only-scene.model}}), [log]({{queryd-train-full-ce-only-scene.log}}) |
| Scene + Inst. | v2t  | {{queryd-train-full-ce-only-scene-inst.v2t}} | {{queryd-train-full-ce-only-scene-inst.params}} | [config]({{queryd-train-full-ce-only-scene-inst.config}}), [model]({{queryd-train-full-ce-only-scene-inst.model}}), [log]({{queryd-train-full-ce-only-scene-inst.log}}) |
| Scene + r2p1d | v2t  | {{queryd-train-full-ce-only-scene-r2p1d.v2t}} | {{queryd-train-full-ce-only-scene-r2p1d.params}} | [config]({{queryd-train-full-ce-only-scene-r2p1d.config}}), [model]({{queryd-train-full-ce-only-scene-r2p1d.model}}), [log]({{queryd-train-full-ce-only-scene-r2p1d.log}}) |
| Scene + Audio | v2t  | {{queryd-train-full-ce-only-scene-audio.v2t}} | {{queryd-train-full-ce-only-scene-audio.params}} | [config]({{queryd-train-full-ce-only-scene-audio.config}}), [model]({{queryd-train-full-ce-only-scene-audio.model}}), [log]({{queryd-train-full-ce-only-scene-audio.log}}) |

We can also study their cumulative effect:

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| Scene    | t2v  | {{queryd-train-full-ce-only-scene.t2v}} | {{queryd-train-full-ce-only-scene.params}} | [config]({{queryd-train-full-ce-only-scene.config}}), [model]({{queryd-train-full-ce-only-scene.model}}), [log]({{queryd-train-full-ce-only-scene.log}}) |
| Prev. + Audio    | t2v  | {{queryd-train-full-ce-only-scene-audio.t2v}} | {{queryd-train-full-ce-only-scene-audio.params}} | [config]({{queryd-train-full-ce-only-scene-audio.config}}), [model]({{queryd-train-full-ce-only-scene-audio.model}}), [log]({{queryd-train-full-ce-only-scene-audio.log}}) |
| Prev. + Inst    | t2v  | {{queryd-train-full-ce-only-scene-audio-inst.t2v}} | {{queryd-train-full-ce-only-scene-audio-inst.params}} | [config]({{queryd-train-full-ce-only-scene-audio-inst.config}}), [model]({{queryd-train-full-ce-only-scene-audio-inst.model}}), [log]({{queryd-train-full-ce-only-scene-audio-inst.log}}) |
| Prev. + R2P1D    | t2v  | {{queryd-train-full-ce-only-scene-audio-inst-r2p1d.t2v}} | {{queryd-train-full-ce-only-scene-audio-inst-r2p1d.params}} | [config]({{queryd-train-full-ce-only-scene-audio-inst-r2p1d.config}}), [model]({{queryd-train-full-ce-only-scene-audio-inst-r2p1d.model}}), [log]({{queryd-train-full-ce-only-scene-audio-inst-r2p1d.log}}) |
| Scene    | v2t  | {{queryd-train-full-ce-only-scene.v2t}} | {{queryd-train-full-ce-only-scene.params}} | [config]({{queryd-train-full-ce-only-scene.config}}), [model]({{queryd-train-full-ce-only-scene.model}}), [log]({{queryd-train-full-ce-only-scene.log}}) |
| Prev. + Audio    | v2t  | {{queryd-train-full-ce-only-scene-audio.v2t}} | {{queryd-train-full-ce-only-scene-audio.params}} | [config]({{queryd-train-full-ce-only-scene-audio.config}}), [model]({{queryd-train-full-ce-only-scene-audio.model}}), [log]({{queryd-train-full-ce-only-scene-audio.log}}) |
| Prev. + Inst.    | v2t  | {{queryd-train-full-ce-only-scene-audio-inst.v2t}} | {{queryd-train-full-ce-only-scene-audio-inst.params}} | [config]({{queryd-train-full-ce-only-scene-audio-inst.config}}), [model]({{queryd-train-full-ce-only-scene-audio-inst.model}}), [log]({{queryd-train-full-ce-only-scene-audio-inst.log}}) |
| Prev. + R2P1D    | v2t  | {{queryd-train-full-ce-only-scene-audio-inst-r2p1d.v2t}} | {{queryd-train-full-ce-only-scene-audio-inst-r2p1d.params}} | [config]({{queryd-train-full-ce-only-scene-audio-inst-r2p1d.config}}), [model]({{queryd-train-full-ce-only-scene-audio-inst-r2p1d.model}}), [log]({{queryd-train-full-ce-only-scene-audio-inst-r2p1d.log}}) |

