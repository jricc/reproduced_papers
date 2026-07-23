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
- providing an exact Qiskit statevector backend for CPU and macOS;
- supporting PneumoniaMNIST as a public alternative dataset;
- documenting the alternative execution path.

The alternative dataset and CPU results are not a reproduction of the paper's
MIMIC-CXR embedding experiments.

The original CC BY-NC-SA 4.0 license is preserved in [LICENSE](LICENSE).
The changes in this adapted copy are distributed under the same license.
