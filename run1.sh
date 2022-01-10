#!/bin/bash
#SBATCH -p gpu3       # 或GPU2、GPU3
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 2         # 一般1个GPU占用一个CPU核即可
#SBATCH --mem 10G     # 申请内存
#SBATCH --gres gpu:1  # 分配一个GPU
#SBATCH -o ./job_report/NTc40-%j.out    # 注意可以修改"job"为与任务相关的内容方便以后查询实验结果
#SBATCH --nodelist wmc-slave-g11 # 可以指定任务在某个节点gx上运行

echo "job begin"
# python3 train+test.py -m Deepfakes -c c40 -wd 1e-05 -bz 32
# python3 train_CNN.py -m Face2Face -c c23 -wd 1e-05 -bz 32
# python3 train_CNN.py -m FaceSwap -c c40 -wd 1e-05 -bz 32
# python3 train+test.py -m NeuralTextures -c c23 -wd 1e-05 -bz 32
python3 train+test.py -m NeuralTextures -c c40 -wd 1e-05 -bz 32 -nw 2
# Deepfakes / Face2Face / FaceSwap / NeuralTextures / All
echo "job end"