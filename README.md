# PIPNet3D
PiPNet3D: Patch-Based Intuitive Prototypes for Interpretable 3D Images Classification

We present PIPNet3D, a part-prototype neural network for volumetric images. 

We applied PIPNet3D to the binary classification of Alzheimer's Disease from 3D structural Magnetic Resonance Imaging (sMRI, T1-MRI). 

We assess the quality of prototypes under a systematic evaluation framework, propose new functionally grounded metrics to evaluate brain prototypes and develop an evaluation scheme to assess their coherency with domain experts.

Classes (clinical cognitive decline level):

- Cognitively Normal (CN)
- Alzheimer's Disease (AD)


**arXiv preprint**: [_"PIPNet3D: Interpretable Detection of Alzheimer in MRI Scans"_](https://arxiv.org/abs/2403.18328)

Accepted at the [iMIMIC](https://imimic-workshop.com) workshop during the [MICCAI-2024](https://conferences.miccai.org/2024/en/) event.

![Overview of PIPNet](https://github.com/desantilisa/PIPNet3D/blob/main/pip3d-overview_v2.pdf)   

Images and labels (cognitive decline level) were collected from the Alzheimer's Disease Neuroimaging Initiative (ADNI) https://adni.loni.usc.edu (data publicity available under request) and preprocessed using data_preprocessing.py functions.

Brain atlas (CerebrA) downloaded from https://nist.mni.mcgill.ca/cerebra/.

Codes adapted from the original [PIPNet](https://github.com/M-Nauta/PIPNet/tree/main)

Training a PIMPNet: main_train_pipnet.py

Test a trained PIMPNet: main_test_pipnet.py

Link to the weights of trained PIMPNet(s) available in "models" folder.
