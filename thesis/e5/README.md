# E5: Teacher-4 Pseudo-Label Preparation

E5 trains four out-of-fold skeleton teachers inside each 8/1/1/1 fold. The pseudo-label target group is stored as the pkl `test` split so PySKL can train with the normal `train` and `val` splits while keeping the target subjects out of training.

Build the required pkls:

```bash
bash thesis/e5/build_pkls.sh
```

Train all 24 teacher configs:

```bash
sbatch thesis/e5/train.sh
```

Useful restricted runs:

```bash
RUN_FOLDS="a" RUN_TEACHERS="t1" RUN_STREAMS="joint limb" sbatch thesis/e5/train.sh
RUN_FOLDS="b c" RUN_TEACHERS="t3 t4" sbatch thesis/e5/train.sh
```
