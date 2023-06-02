# for i in {2..2}; do
#     python main.py --config "algorithms/ERM/configs/CIFAR10.json" --exp_idx $i --gpu_idx "0"
# done

# taskset -c "51" python main.py --config "algorithms/mDSDI/configs/PACS_photo.json" --exp_idx $i --gpu_idx "1"
# python main.py --config "algorithms/CVAE/configs/MNIST.json" --exp_idx "1" --gpu_idx "0"
# python main.py --config "algorithms/Flows/configs/MNIST.json" --exp_idx "1" --gpu_idx "0"

rm -r algorithms/VAE/results/checkpoints/*
rm -r algorithms/VAE/results/logs/*
rm -r algorithms/VAE/results/plots/*
rm -r algorithms/VAE/results/tensorboards/*

# python utils/tSNE_plot.py --plotdir "/home/ubuntu/gradensity_inference/algorithms/SM/results/plots/MNIST_1/"

# tensorboard --logdir "/data/habui/gradensity_inference/algorithms/SM/results/tensorboards/Rotated_75_MNIST_0"
# tensorboard dev upload --logdir "/data/habui/gradensity_inference/algorithms/SM/results/tensorboards/Rotated_75_MNIST_0"

# black -l 119 ./
# isort -l 119 --lai 2 -m 3 --up --sd 'FIRSTPARTY' -n --fgw 0 --tc ./
