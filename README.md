# density_softmax
## <a name="demo"></a> Quick Demo
Run [this Google Colab](https://colab.research.google.com/drive/1fdsAW_J4WKBFSTa3Hc2EDo_ZbjsoSle7?usp=sharing).

or

notebook in `demo/density_softmax.ipynyb`

or 

python file (full comparision, install prerequisite packages first to import library):
```sh
python demo/demo.py
```

## <a name="guideline"></a> Benchmark Guideline
### <a name="prepare"></a> To prepare:
Install prerequisite packages:
```sh
pip install "git+https://github.com/google/uncertainty-baselines.git#egg=uncertainty_baselines"
```

and

```sh
bash setup.sh
```

### <a name="experiments"></a> To run experiments:
```sh
python <method_file> --data_dir=<data_path>  --output_dir=<output_path> --use_gpu="True" --num_cores="1" 
```
where the parameters are the following:
- `<method_file>`: file stored the code of method. E.g., `<method_file> = baselines/cifar/density_softmax.py`
- `<data_path>`: path stored the dataset. E.g., `<data_path> = "tmp/tensorflow_datasets"`
- `<output_path>`: path to store outputs of the model. E.g., `<output_path> = "tmp/cifar10/density_softmax"`

## References
Based on code of: ["Uncertainty Baselines: Benchmarks for uncertainty & robustness in deep learning"](https://github.com/google/uncertainty-baselines)
> Z. Nado, N. Band, M. Collier, J. Djolonga, M. Dusenberry,
> S. Farquhar, A. Filos, M. Havasi, R. Jenatton, G.
> Jerfel, J. Liu, Z. Mariet, J. Nixon, S. Padhy, J. Ren, T.
> Rudner, Y. Wen, F. Wenzel, K. Murphy, D. Sculley, B.
> Lakshminarayanan, J. Snoek, Y. Gal, and D. Tran.
> [Uncertainty Baselines:  Benchmarks for uncertainty & robustness in deep learning](https://arxiv.org/abs/2106.04015),
> _arXiv preprint arXiv:2106.04015_, 2021.

## License
This source code is released under the Apache-2.0 license, included [here](LICENSE).