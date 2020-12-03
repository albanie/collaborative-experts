This repo provides code for TeachText.

**Requirements:** The code assumes PyTorch 1.4 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.

**MSRVTT Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{msrvtt-train-full-ce.t2v}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-adam.t2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-adam.model}}), [log]({{msrvtt-train-gpt2-xl-finetuned-adam.log}}) |
| TeachText - CE    | Full  | t2v  | {{msrvtt-train-ce-intra-mte.t2v}} | [config]({{msrvtt-train-ce-intra-mte.config}}), [model]({{msrvtt-train-ce-intra-mte.model}}), [log]({{msrvtt-train-ce-intra-mte.log}}) |
| TeachText - CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-mte-adam.t2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.model}}), [log]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.log}}) |

**Denoising results on MSRVTT**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-denoising-adam.t2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.model}}), [log]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.log}}) |
| TeachText - CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.t2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.model}}), [log]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.log}}) |
