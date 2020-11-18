# Langevin Dynamics and Score Matching

## How to?

### Train

```
python3 main.py config/ring.yml train
```

This command produces the model (learnt score function): `ring.pth`

### Test

```
python3 main.py config/ring.yml test
```

This command produces a list of frames at the stages of sampling under `ring/` and an animation `ring.gif` made from them.

## Demo (with carefully tuned parameters)

Left: sample from the real pixel distribution
Right: sample by Langevin dynamics based on the score function learnt by denoising score matching.

![ring.gif](./ring.gif)

![waddles.gif](./waddles.gif)

![stewie.gif](./stewie.gif)

### Reference

- A good [lecture](https://youtu.be/3-KzIjoFJy4) on Langevin dynamics for sampling.
- A [recent paper](https://arxiv.org/pdf/1907.05600.pdf) combining Langevin dynamics with denosing score matching with a more complicated multi-level pertubation scheme.
- A [paper](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) introducing the denoising score matching.
