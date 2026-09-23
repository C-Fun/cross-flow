监控命令清单

1. 看作业队列状态(含依赖关系)
`squeue -u $USER -o "%.10i %.18j %.8T %.10M %.6D %.18R %E"`
%T=状态(RUNNING/PENDING),%M=已运行时间,%R=节点或等待原因,%E=依赖(afterany)。
先用这条确认那 4 个作业分别是什么——正常应该是一个在跑、一个 pending 挂在前一个后面;若有多余的重复训练作业,scancel <jobid> 掉。

2. 实时看训练日志(最主要的监控手段)
sbatch 日志写在你提交作业的目录,文件名 crossflow-<jobid>.out:
`tail -f crossflow-433774.out`
能看到 MIOpen tuning →之后 step N/400000 | loss_cf ... | lpips ... | img/s 的训练日志、采样、FID。

3. 看 GPU 是否真的在算
先从 squeue 拿到节点名(比如 k005-003),然后:
`srun --jobid=433774 --overlap amd-smi | head -20`
GFX-Uti 应该是高占用、显存用起来了。

4. wandb 网页(你开了在线模式,最省事)
浏览器打开 wandb 项目 crossflow,直接看 loss/lpips/FID 曲线和采样图,不用登服务器。

5. 停止当前任务链：`scancel -u $USER --name=crossflow`

6. 查询排队预估时间：`squeue --start -j <jobid>`