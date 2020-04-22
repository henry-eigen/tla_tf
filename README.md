# tla_tf



This is an implementation of TLA from:
    [Metric Learning for Adversarial Robustness](https://arxiv.org/pdf/1909.00900.pdf)

The loss is built so that the network has a single input. A batch of clean images and their adversaries is passed to the network in the form:
    [X<sub>1</sub> X<sub>2</sub> X<sub>3</sub>... X<sub>n</sub> X<sub>1</sub>' X<sub>2</sub>' X<sub>3</sub>'... X<sub>n</sub>']

where X<sub>i</sub>' is the adversarially crafted from X<sub>i</sub>
