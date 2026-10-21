# HPC3 operations

This project follows the course HPC guide. It is designed for off-campus work:
connect to the SJSU VPN, SSH to `coe-hpc1.sjsu.edu`, then hop to `coe-hpc3`.
Never store a username, password, or real group directory in this repository.

## Storage and cache policy

- Use the course-group `/data` directory for datasets and reusable model/cache
  artifacts.
- Use the course-group `/scratch` directory for generated predictions, logs,
  temporary files, and Slurm outputs.
- Keep the home directory small; redirect `HF_HOME` and `TORCH_HOME` away from
  it.
- Keep dataset files, checkpoints, raw predictions, and outputs out of Git.

The exact current-term group paths are not assumed. Set them only through a
local environment file based on `configs/hpc.env.example` after verifying the
assigned directory.

## Execution policy

1. Use the head node only for Git, staging, environment setup, and job
   submission.
2. Inspect modules and available partitions with `module avail` and `sinfo`.
3. Perform a 20-image smoke test through an interactive GPU allocation.
4. Submit the pilot and full experiments through Slurm; record every job ID.
5. Stage model weights and packages before jobs where possible. Compute nodes
   may lack normal outbound internet access.

## Planned validation commands

```bash
module avail
sinfo
conda info --envs
srun -p "$SLURM_PARTITION" --pty /bin/bash
nvidia-smi
python -m torch.utils.collect_env
```

The course guide gives `gpuqs` and `nsfqs` as examples, but availability and
the GPU generation allocated to a particular job must be observed, not assumed.

## Source guide

The operational details above are derived from the professor's
[`hpc3.md`](https://github.com/lkk688/DeepDataMiningLearning/blob/main/docs/source/hpc3.md).
