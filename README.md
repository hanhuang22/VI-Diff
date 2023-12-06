# VI-Diff: Unpaired Visible-Infrared Translation Diffusion Model for Single Modality Labeled Visible-Infrared Person Re-identification

**This is experiment code. Will clean up later.**

**datasets**: put visible/infrared images to trainA/trainB

    for sysu-mm01: rename images in format like 'cam1_0001_0002.jpg'

**training/testing**: 
    
    use bash scripts to train/test.

    regdb: extract Hed edge, and modify the edge image path in 'guided_diffusion/image_datasets.py';
           edge image name same as original image.

    modify 'process_gpu_dict' in 'guided_diffusion/dist_util.py' to use your GPU cluster.


**Citations**
    
    @misc{huang2023vidiff,
        title={VI-Diff: Unpaired Visible-Infrared Translation Diffusion Model for Single Modality Labeled Visible-Infrared Person Re-identification}, 
        author={Han Huang and Yan Huang and Liang Wang},
        year={2023},
        eprint={2310.04122},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

