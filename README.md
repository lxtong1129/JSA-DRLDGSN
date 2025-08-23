# JAS-DRLDGSN

This is the source code of the paper: "Multi-Agent Reinforcement Learning for Resource Allocation in NOMA-Enhanced Aerial Edge Computing Networks". A brief introduction of this work is as follows:
> The advantages of Unmanned Aerial Vehicles (UAVs) in terms of maneuverability and line-of-sight communication have made Aerial Edge Computing (AEC) a  promising solution for processing computationally intensive tasks. However, constrained computational resources of UAVs and the complexity of multi-UAV coordination pose significant challenges in designing efficient trajectory optimization and power allocation strategies to enhance user service quality. To address this issue, we construct an AEC architecture assisted by non-orthogonal multiple access (NOMA) and a deep reinforcement learning (DRL) algorithm based on dynamic Gaussian mixture and sharing networks (DRL-DGSN). By leveraging NOMA's successive interference cancellation technology, DRL-DGSN simultaneously optimizes user association, UAV power allocation, and trajectory design to maximize system throughput. Specifically. First, DRL-DGSN employs a dynamic user association algorithm based on Gaussian mixture model, which achieves capacity-aware uniform clustering through probabilistic modeling combined with cluster capacity constraints, effectively preventing UAV overload. Second, DRL-DGSN utilizes a multi-agent DRL framework with a dueling network architecture and double deep Q-networks. By integrating a shared network, agents can efficiently share experiences, enabling simultaneous optimization of multi-UAV cooperative trajectory and power allocation while reducing Q-value overestimation and enhancing training efficiency. Finally, extensive experiments validate the superiority and effectiveness of DRL-DGSN across various scenarios.

## Dependencies
- python 3.7.x
- tensorflow 2.x

## Usage
To start our project files: run
`main.py`

## Citation
This work is part of ongoing research on aerial edge computing. The corresponding paper is currently under review at Journal of Systems Architecture. If this code package has contributed to your work, please cite the original paper. Thanks.

```
@article{bib01-JSA2024-DRLDGSN,
  title={Multi-Agent Reinforcement Learning for Resource Allocation in NOMA-Enhanced Aerial Edge Computing Networks},
  author={Zhang, Longxin and Lu, Xiaotong and Zhang, Yanfen and Cao, Buqing and Li, Keqin},
  journal={Journal of Systems Architecture},
  year={2024},
  note={Under review}
}
