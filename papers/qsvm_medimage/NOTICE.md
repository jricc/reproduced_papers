# Attribution and adaptation notice

This directory is an adapted copy of the original QML-MedImage repository:

- Original source: <https://github.com/sebasmos/qml-medimage>
- Original paper: *Quantum Kernel Advantage over Classical Collapse in Medical
  Foundation Model Embeddings*, <https://arxiv.org/abs/2604.24597>
- Reproduction collection: <https://github.com/merlinquantum/reproduced_papers>

The original authors retain attribution for the original code and scientific
work. This adaptation is maintained in a fork of `merlinquantum/reproduced_papers`.

Local modifications are intentionally limited to:

- making NVIDIA/cuQuantum dependencies optional;
- making Torch optional for the CPU-only QSVM import path;
- providing an exact Qiskit statevector backend for CPU and macOS;
- supporting PneumoniaMNIST as a public alternative dataset;
- providing a separate MerLin 0.4 photonic fidelity-kernel adaptation;
- documenting the alternative execution path.

The alternative dataset and CPU results are not a reproduction of the paper's
MIMIC-CXR embedding experiments.

MerLin 0.4 requires scikit-learn 1.7.2 or newer. The combined local environment
therefore differs from the original scikit-learn 1.6.1 pin; generated artifacts
must record the effective dependency versions.

The original CC BY-NC-SA 4.0 license is preserved in [LICENSE](LICENSE).
The changes in this adapted copy are distributed under the same license.

## PneumoniaMNIST

<https://zenodo.org/records/10519652>

```bibtex
@article{medmnistv2,
    title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
    author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
    journal={Scientific Data},
    volume={10},
    number={1},
    pages={41},
    year={2023},
    publisher={Nature Publishing Group UK London}
}
@inproceedings{medmnistv1,
    title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
    author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
    booktitle={IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
    pages={191--195},
    year={2021}
}
```
The MedMNIST dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), except DermaMNIST under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

The code is under Apache-2.0 License.
