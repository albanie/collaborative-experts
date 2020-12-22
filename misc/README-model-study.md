### MODEL study on MSRVTT

**Importance of the model**:

| Model | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| HowTo100m S3D | t2v  | {{msrvtt-train-full-mnnet.geomt2v}} | {{msrvtt-train-full-mnnet.params}} | [config]({{msrvtt-train-full-mnnet.config}}), [model]({{msrvtt-train-full-mnnet.model}}), [log]({{msrvtt-train-full-mnnet.log}}) |
| CE - P,CG | t2v  | {{msrvtt-train-full-moee.geomt2v}} | {{msrvtt-train-full-moee.params}} | [config]({{msrvtt-train-full-moee.config}}), [model]({{msrvtt-train-full-moee.model}}), [log]({{msrvtt-train-full-moee.log}}) |
| CE    | t2v  | {{msrvtt-train-full-ce.geomt2v}} | {{msrvtt-train-full-ce.params}} |[config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| HowTo100m S3D | v2t  | {{msrvtt-train-full-mnnet.geomv2t}} | {{msrvtt-train-full-mnnet.params}} | [config]({{msrvtt-train-full-mnnet.config}}), [model]({{msrvtt-train-full-mnnet.model}}), [log]({{msrvtt-train-full-mnnet.log}}) |
| CE - P,CG | v2t  | {{msrvtt-train-full-moee.geomv2t}} | {{msrvtt-train-full-moee.params}} | [config]({{msrvtt-train-full-moee.config}}), [model]({{msrvtt-train-full-moee.model}}), [log]({{msrvtt-train-full-moee.log}}) |
| CE    | v2t  | {{msrvtt-train-full-ce.geomv2t}} | {{msrvtt-train-full-ce.params}} |[config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
