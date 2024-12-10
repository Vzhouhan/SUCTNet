<img align="right" src="https://ars.els-cdn.com/content/image/X01438166.jpg" width="290" height="350"/>  

# Supervised-unsupervised combined transformer for spectral compressive imaging reconstruction  

* ### Abstract
To solve the low spatial and/or temporal resolution problem which the conventional hyperspectral cameras often suffer from, spectral compressive imaging systems (SCI) have attracted more attention recently. Recovering a hyperspectral image (HSI) from its corresponding 2D coded image is an ill-posed inverse problem, and learning accurate prior from HSI and 2D coded image is essential to solve this inverse problem. Existing methods only use supervised networks that focus on learning generalized prior from training datasets, or only use unsupervised networks that focus on learning specific prior from 2D coded image, resulting in the inability to learn both generalized and specific priors. Also, when learning the priors, existing methods cannot simultaneously give consideration to both global and local scales, as well as both spatial and spectral dimensions. To cope with this problem, in this paper, we propose a Supervised-Unsupervised Combined Transformer Network (SUCTNet) composed by a supervised Spatio-spectral Transformer network (SSTNet) and an Unsupervised Multi-level Feature Refinement network (UMFRNet). Specifically, we first develop the SSTNet to learn generalized prior and obtain a preliminary HSI. In SSTNet, the proposed spatial encoding and spectral decoding network architecture enables it to simultaneously consider both spatial and spectral dimensions, and a proposed Global and Local Multi head Self Attention block (GL-MSA) enables it simultaneously to consider both global and local scales. Then, the preliminary HSI is fed into the proposed UMFRNet to learn specific prior and obtain the target HSI. In UMFRNet, a proposed multi-level feature refinement mechanism and the physical imaging model of SCI are used to improve reconstruction accuracy and generalization performance. Extensive experiments show that our method significantly outperforms state-of-the-art (SOTA) methods on simulated and real datasets.  

# Flowchart
![Performance](https://github.com/Vzhouhan/SUCTNet/blob/main/SSTNet.png)  
![Performance](https://github.com/Vzhouhan/SUCTNet/blob/main/UMFRNet.png)
# Result presentation
![Performance](https://github.com/Vzhouhan/SUCTNet/blob/main/Simu%20Scene5.png) 
![Performance](https://github.com/Vzhouhan/SUCTNet/blob/main/Simu%20Scene10.png) 
![Performance](https://github.com/XWangBin/CST-Net/blob/main/IMGs/result3.png) 
![Performance](https://github.com/XWangBin/CST-Net/blob/main/IMGs/result4.png) 

# Datasets
[`dataset download, model train from here`](https://github.com/yhao-z/T2LR-Net)

# Note
For any questions, feel free to email me at wangb@nim.ac.cn.  
If you find our work useful in your research, please cite our paper ^.^

# Acknowledgments
[`SLR-Net`](https://github.com/Keziwen/SLR-Net),[`L+S-Net`],[`DUS-Net`](https://github.com/yhao-z/DUS-Net),[`T2LR-Net`](https://github.com/yhao-z/T2LR-Net),

