# HPC3 operations

This project follows the course HPC guide. It is designed for off-campus work:
connect to the SJSU VPN, SSH to `coe-hpc1.sjsu.edu`, then hop to `coe-hpc3`.
Never store a username, password, or real group directory in this repository.

## Storage and cache policy

- Read datasets only from the course-group `/data` directory.
- Use a `scratch/` directory inside your own student workspace for generated
  predictions, logs, temporary files, and Slurm outputs.
- Keep the home directory small; redirect `HF_HOME` and `TORCH_HOME` away from
  it.
- Keep dataset files, checkpoints, raw predictions, and outputs out of Git.

Create a personal virtual environment over the course Python with
`--system-site-packages`, then set its path and all writable output paths in a
local environment file based on `configs/hpc.env.example`.

## Execution policy

1. Use the head node only for Git, staging, environment setup, and job
   submission.
2. Request **exactly one GPU** per job. This course's GPU policy prohibits
   multi-GPU reservations for this project.
3. Inspect modules and available partitions with `module avail` and `sinfo`.
4. Submit the 20-image smoke test as a short, single-GPU batch job, then
   inspect its log and release resources automatically at job completion.
5. Submit the pilot and full experiments through Slurm with realistic time
   limits; record every job ID. Do not hold an allocation while debugging on
   CPU, transferring files, or waiting for input.
6. Check `nvidia-smi` during early runs to verify meaningful GPU utilization;
   end jobs that are stalled or accidentally CPU-bound.
7. Stage model weights and packages before jobs where possible. Compute nodes
   may lack normal outbound internet access.

## Planned validation commands

```bash
module avail
sinfo
python /data/cmpe249-fa26/examples/check_env.py  # inside the batch pilot
python -m torch.utils.collect_env                 # inside the batch pilot
```

Before a GPU pilot, submit `scripts/hpc/preflight.sbatch`. It is a 15-minute,
CPU-only batch validation of the exact COCO and KITTI paths used by this
project.

The observed class-accessible starting point is `gpuqs`; submit a one-P100
request first. Do not use the faculty `condo` or NSF partitions unless the
instructor explicitly authorizes them.

## Source guide

The operational details above are derived from the professor's
[`hpc3.md`](https://github.com/lkk688/DeepDataMiningLearning/blob/main/docs/source/hpc3.md).
