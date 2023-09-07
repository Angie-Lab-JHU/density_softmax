# python baselines/cifar/rank1_bnn.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/rank1_bnn_cifar10" --use_gpu="True" --num_cores="1" --seed="2"
python baselines/imagenet/density_softmax.py --data_dir="/root/tensorflow_datasets" --output_dir="out/tmpl" --use_gpu="True" --num_cores="1"
# python baselines/imagenet/batchensemble.py --data_dir="/root/tensorflow_datasets" --output_dir="out/tmpl" --use_gpu="True" --num_cores="1"

# python baselines/cifar/ensemble_copy.py --data_dir="tmp/tensorflow_datasets" --corruption_type="brightness" --severity="4" --output_dir="utils/out/ensemble_cifar10/model" --use_gpu="True" --num_cores="1"

# python baselines/cifar/variational_inference.py --data_dir="tmp/tensorflow_datasets" --output_dir="utils/out/variational_inference_cifar10/model" --use_gpu="True" --num_cores="1"

# python baselines/cifar/sngp.py --data_dir="tmp/tensorflow_datasets" --output_dir="utils/out/sngp_cifar10/model" --use_gpu="True" --num_cores="1"

# python baselines/imagenet/rank1_bnn.py --data_dir="/root/tensorflow_datasets" --output_dir="tmp/rank1_bnn_imagenet" --use_gpu="True" --num_cores="1"

# python baselines/imagenet/test.py --data_dir="/root/tensorflow_datasets" --output_dir="utils/out/imagenet/model_imagenet" --use_gpu="True" --num_cores="1"

# python baselines/cifar/variational_inference.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/variational_inference_cifar10" --use_gpu="True" --num_cores="1"


# python baselines/cifar/variational_inference.py --data_dir="tmp/tensorflow_datasets" --output_dir="tmp/variational_inference_cifar100" --use_gpu="True" --num_cores="1"


