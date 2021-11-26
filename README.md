# Explainable Face Image Quality

***26.11.2021*** _start readme_


## QMagFace: Simple and Accurate Quality-Aware Face Recognition

* [Research Paper](https://arxiv.org/abs/2110.11001) todo
* [Implementation on ArcFace](face_image_quality.py) todo



## Table of Contents 

- [Abstract](#abstract)
- [Results](#results)
- [Installation](#installation)
- [Citing](#citing)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Abstract

<img src="visualization with explanation.png" height="470" align="right">

Face recognition systems have to deal with large variabilities (such as different poses, illuminations, and expressions) that might lead to incorrect matching decisions. These variabilities can be measured in terms of face image quality which is defined over the utility of a sample for recognition. Previous works on face recognition either do not employ this valuable information or make use of noninherently fit quality estimates. In this work, we propose a simple and effective face recognition solution (QMag- Face) that combines a quality-aware comparison score with a recognition model based on a magnitude-aware angular margin loss. The proposed approach includes modelspecific face image qualities in the comparison process to enhance the recognition performance under unconstrained circumstances. Exploiting the linearity between the qualities and their comparison scores induced by the utilized loss, our quality-aware comparison function is simple and highly generalizable. The experiments conducted on several face recognition databases and benchmarks demonstrate that the introduced quality-awareness leads to consistent improvements in the recognition performance. Moreover, the proposed QMagFace approach performs especially well under challenging circumstances, such as crosspose, cross-age, or cross-quality. Consequently, it leads to state-of-the-art performances on several face recognition benchmarks, such as 98.50% on AgeDB, 83.97% on XQLFQ, and 98.74% on CFP-FP.





## Results

The proposed approach is analysed in three steps. 
First, we report the performance of QMagFace on six face recognition benchmarks against ten recent state-of-the-art methods in image- and video-based recognition tasks to provide a comprehensive comparison with state-of-the-art. 
Second, we investigate the face recognition performance of QMagFace over a wide FMR range to show its suitability for a wide variety of applications and to demonstrate that the quality-aware comparison score constantly enhances the recognition performance. 
Third, we analyse the optimal quality weight over a wide threshold range to demonstrate the robustness of the training process and the generalizability of the proposed approach.

In the following, we will only show some results. For more details and dicussions, please take a look at the paper.

<img src="Table_Benchmarks.png " height = "450" align = "right" > 

**Performance on face recognition benchmarks** - The face recognition performance on the four benchmarks is reported in terms of benchmark accuracy (%). The
highest performance is marked bold. The proposed approach, QMagFace-100, achieves state-of-the-art face recognition performance, especially in cross-age (AgeDB), cross-pose (CFP-FP), and cross-quality (XQLFW) scenarios.
Since the FIQ captures these challenging conditions and the quality values represent the utility of the images for our specific network, the proposed quality-aware comparison score can specifically address the circumstance and their effect on the network. 
Consequently, it performs highly accurate in the cross-age, cross-pose, and cross-quality scenarios and achieves state-of-the-art performances.



**Face recognition performance over a wide range of FMRs** - The face recognition performance is reported in terms of FNMR [%] over a wide range of FMRs. The MagFace and the proposed QMagFace approach are compared for three backbone architectures on three databases. The better values between both approaches are highlighted in
bold. In general, the proposed quality-aware solutions constantly improve the performance, often by a large margin. This is especially true for QMagFace based on the iResNet-100 backbone.

<img src="Table_QualityAwareness.png" height = "450" > 

**Robustness analysis** - The optimal quality weight for different decision thresholds is reported on four databases. 
Training on different databases lead to similar linear solutions for the quality-weighting function. The results demonstrate that (a) the choice of a linear function
is justified and (b) that the learned models have a high generalizability since the quality-weighting function trained on one database is very
similar to the optimal functions of the others.

<img src="OptimalQualityFunctions.png"  > 


## Installation
to do Malte

only QMagFace-100




## Citing

If you use this code, please cite the following paper.

todo

```
@article{DBLP:journals/corr/abs-2110-11001,
  author    = {Philipp Terh{\"{o}}rst and
               Marco Huber and
               Naser Damer and
               Florian Kirchbuchner and
               Kiran Raja and
               Arjan Kuijper},
  title     = {Pixel-Level Face Image Quality Assessment for Explainable Face Recognition},
  journal   = {CoRR},
  volume    = {abs/2110.11001},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.11001},
  eprinttype = {arXiv},
  eprint    = {2110.11001},
  timestamp = {Thu, 28 Oct 2021 15:25:31 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-11001.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

If you make use of our implementation based on MagFace, please additionally cite the original ![MagFace module](https://github.com/IrvingMeng/MagFace).

## Acknowledgement

This research work has been funded by the German Federal Ministry of Education and Research and the Hessen State Ministry for Higher Education, Research and the Arts within their joint support of the National Research Center for Applied Cybersecurity ATHENE.
Portions of the research in this paper use the FERET database of facial images collected under the FERET program, sponsored by the DOD Counterdrug Technology Development Program Office.
This work was carried out during the tenure of an ERCIM ’Alain Bensoussan‘ Fellowship Programme.

## License 

This project is licensed under the terms of the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt

