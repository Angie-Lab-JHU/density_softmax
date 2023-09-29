# python baselines/cifar/rank1_bnn.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/rank1_bnn_cifar10" --use_gpu="True" --num_cores="1" --seed="2"
# python baselines/cifar/density_softmax.py --data_dir="tmp/tensorflow_datasets" --output_dir="out/tmpl" --use_gpu="True" --num_cores="1"
# python baselines/imagenet/density_softmax.py --data_dir="/root/tensorflow_datasets" --output_dir="checkpoints/density_softmax/imagenet/tmp" --use_gpu="True" --num_cores="1"
# python baselines/cifar/test.py --data_dir="tmp/tensorflow_datasets" --output_dir="checkpoints/density_softmax/cifar10/model4" --use_gpu="True" --num_cores="1" --seed="1"
# python baselines/cifar/test_deterministic.py --data_dir="tmp/tensorflow_datasets" --output_dir="checkpoints/deterministic/cifar10/model3" --use_gpu="True" --num_cores="1" --seed="3"

# taskset --cpu-list 0-7 python baselines/cifar/deterministic.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/cifar/dropout.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/cifar/rank1_bnn.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/cifar/variational_inference.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/cifar/batchensemble.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/cifar/heteroscedastic.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/cifar/posterior_network.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/cifar/sngp.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/cifar/mimo.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
taskset --cpu-list 0-7 python baselines/cifar/density_softmax.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"

# taskset --cpu-list 0-7 python baselines/imagenet/deterministic.py --data_dir="/root/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/imagenet/batchensemble.py --data_dir="/root/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/imagenet/dropout.py --data_dir="/root/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/imagenet/mimo.py --data_dir="/root/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/imagenet/rank1_bnn.py --data_dir="/root/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"
# taskset --cpu-list 0-7 python baselines/imagenet/sngp.py --data_dir="/root/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"

# taskset --cpu-list 0-7 python baselines/cifar/deterministic.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/tmp" --use_gpu="True" --num_cores="1" --seed="1"

# python baselines/cifar/posterior_network.py --data_dir="tmp/tensorflow_datasets" --output_dir="checkpoints/posterior_network/cifar100/model1" --use_gpu="True" --num_cores="1" --seed="1"
# python baselines/cifar/mimo.py --data_dir="tmp/tensorflow_datasets" --output_dir="checkpoints/mimo/cifar100/model1" --use_gpu="True" --num_cores="1" --seed="1"

# python baselines/cifar/test_cifar_100.py --data_dir="tmp/tensorflow_datasets" --output_dir="checkpoints/density_softmax/cifar10/model1" --use_gpu="True" --num_cores="1" --seed="1"

# python baselines/cifar/ensemble_copy.py --data_dir="tmp/tensorflow_datasets" --corruption_type="brightness" --severity="4" --output_dir="utils/out/ensemble_cifar10/model" --use_gpu="True" --num_cores="1"

# python baselines/cifar/variational_inference.py --data_dir="tmp/tensorflow_datasets" --output_dir="utils/out/variational_inference_cifar10/model" --use_gpu="True" --num_cores="1"

# python baselines/cifar/sngp.py --data_dir="tmp/tensorflow_datasets" --output_dir="utils/out/sngp_cifar10/model" --use_gpu="True" --num_cores="1"

# python baselines/imagenet/rank1_bnn.py --data_dir="/root/tensorflow_datasets" --output_dir="tmp/rank1_bnn_imagenet" --use_gpu="True" --num_cores="1"

# python baselines/imagenet/test.py --data_dir="/root/tensorflow_datasets" --corruption_type="brightness" --severity="4" --output_dir="checkpoints/density_softmax/imagenet/model1" --use_gpu="True" --num_cores="1"

# python baselines/cifar/variational_inference.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/variational_inference_cifar10" --use_gpu="True" --num_cores="1"


# python baselines/cifar/variational_inference.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/variational_inference_cifar100" --use_gpu="True" --num_cores="1"


