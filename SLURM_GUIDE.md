# SLURM Training Guide for Cross-View Localization

This guide helps you run training on SLURM clusters with NVIDIA 5090 GPUs.

**定位方式**: 单向定位（前视图 → 卫星图）- 基于前视图的提示，在卫星图上进行定位。

## 📋 Quick Start

### 1. Single GPU (5090 has 32GB, should fit the model)

```bash
# DDP (simpler, recommended for single GPU)
sbatch scripts/slurm_train_ddp.sh configs/default.yaml

# Accelerate (more features, but overkill for single GPU)
sbatch scripts/slurm_train_accelerate.sh configs/default.yaml
```

### 2. Multi-GPU (2-8 GPUs)

**Edit the script first** to request more GPUs:

```bash
# Open the script
vim scripts/slurm_train_accelerate.sh

# Change this line:
#SBATCH --gres=gpu:1
# To:
#SBATCH --gres=gpu:4  # For 4 GPUs

# Then submit
sbatch scripts/slurm_train_accelerate.sh configs/default.yaml
```

---

## 🔧 Configuration

### Before First Run

1. **Check your cluster's partition name:**
   ```bash
   sinfo  # List all partitions
   ```
   
   Update the script if needed:
   ```bash
   #SBATCH --partition=gpu  # Change 'gpu' to your partition name
   ```

2. **Verify conda environment:**
   ```bash
   conda activate filtre
   pip install -r requirements.txt
   ```

3. **Test locally first (if possible):**
   ```bash
   # On a compute node with GPU
   srun --gres=gpu:1 --pty bash
   conda activate filtre
   python train_accelerate.py --config configs/default.yaml
   ```

---

## 📊 SLURM Commands Cheat Sheet

### Submit Jobs

```bash
# Submit DDP training
sbatch scripts/slurm_train_ddp.sh configs/default.yaml

# Submit Accelerate training with custom config
sbatch scripts/slurm_train_accelerate.sh configs/my_config.yaml

# Submit with specific GPU count (edit script first)
sbatch scripts/slurm_train_accelerate.sh
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check detailed job info
scontrol show job <JOB_ID>

# Watch job status in real-time
watch -n 1 squeue -u $USER

# Check GPU usage on your job
srun --jobid=<JOB_ID> --pty nvidia-smi
```

### View Logs


# View output log (updates in real-time)
tail -f logs/slurm_<JOB_ID>.out


### Cancel Jobs

# Cancel a specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel jobs by name
scancel --name=cvloc_accel

---

## 🔍 Troubleshooting

### Job Pending Forever

```bash
# Check why job is pending
squeue -u $USER --start

# Common reasons:
# - No available GPUs: wait or reduce --gres=gpu:N
# - Wrong partition: check with 'sinfo' and update script
# - Insufficient memory: reduce --mem in script
```


### Job Fails Immediately

```bash
# Check error log
cat logs/slurm_<JOB_ID>.err

# Common issues:
# 1. Conda environment not activated
#    → Check 'source ~/.bashrc' and 'conda activate filtre' in script
# 2. Missing dependencies
#    → Run: conda activate filtre && pip install -r requirements.txt
# 3. Wrong paths
#    → Ensure data paths in configs/default.yaml are correct
```

### NCCL Timeout (Multi-GPU)

```bash
# Add to script before training command:
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800  # 30 minutes

# Or reduce number of GPUs
#SBATCH --gres=gpu:2  # Instead of 4 or 8
```

---

## 📈 Monitoring Training

### TensorBoard (Accelerate only)

```bash
# On login node or compute node with port forwarding
tensorboard --logdir=output/cross_view_localization --port=6006

```

### Check Training Progress

```bash
# Watch loss in real-time
tail -f logs/slurm_<JOB_ID>.out | grep "loss"

# Check GPU memory usage
srun --jobid=<JOB_ID> --pty nvidia-smi

# Check checkpoint files
ls -lh output/best/
ls -lh output/epoch_*/
```

---

## 🚀 Advanced: Multi-Node Training

If you need to scale beyond 8 GPUs on a single node:

```bash
#SBATCH --nodes=2              # 2 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8           # 8 GPUs per node = 16 total

# Use Accelerate (DDP doesn't support multi-node easily)
sbatch scripts/slurm_train_accelerate.sh
```

## 📝 Example Workflow

```bash
# 1. Test on single GPU first
sbatch scripts/slurm_train_ddp.sh configs/default.yaml
# Wait for job ID, e.g., 12345

# 2. Monitor
tail -f logs/slurm_12345.out

# 3. If OOM, reduce batch size and resubmit
vim configs/default.yaml  # Change batch_size: 4
sbatch scripts/slurm_train_ddp.sh configs/default.yaml

# 4. If successful, scale to multi-GPU
vim scripts/slurm_train_accelerate.sh  # Change --gres=gpu:4
sbatch scripts/slurm_train_accelerate.sh configs/default.yaml

# 5. Check TensorBoard
tensorboard --logdir=output/cross_view_localization
```

---

## 📞 Getting Help

1. **Check SLURM logs**: `logs/slurm_<JOB_ID>.{out,err}`
2. **Check training logs**: `output/logs/train.log`
3. **Test locally**: `srun --gres=gpu:1 --pty bash`
4. **Contact cluster admin**: For partition names, GPU availability, etc.

---

## ✅ Pre-Flight Checklist

Before submitting your first job:

- [ ] Conda environment `filtre` is set up
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Data paths in `configs/default.yaml` are correct
- [ ] SLURM partition name is correct in scripts
- [ ] `logs/` directory exists: `mkdir -p logs`
- [ ] Tested script syntax: `bash -n scripts/slurm_train_ddp.sh`

Good luck with your training! 🎉
