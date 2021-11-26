# Explainable Face Image Quality

***26.11.2021*** _start readme_


## QMagFace: Simple and Accurate Quality-Aware Face Recognition

* [Research Paper](https://arxiv.org/abs/2110.11001) todo
* [Implementation on ArcFace](face_image_quality.py) todo



## Table of Contents 

- [Abstract](#abstract)
- [Key Points](#key-points)
- [Results](#results)
- [Installation](#installation)
- [Citing](#citing)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Abstract

<img src="visualization with explanation.png" height="470" align="right">

Face recognition systems have to deal with large variabilities (such as different poses, illuminations, and expressions) that might lead to incorrect matching decisions. These variabilities can be measured in terms of face image quality which is defined over the utility of a sample for recognition. Previous works on face recognition either do not employ this valuable information or make use of noninherently fit quality estimates. In this work, we propose a simple and effective face recognition solution (QMag- Face) that combines a quality-aware comparison score with a recognition model based on a magnitude-aware angular margin loss. The proposed approach includes modelspecific face image qualities in the comparison process to enhance the recognition performance under unconstrained circumstances. Exploiting the linearity between the qualities and their comparison scores induced by the utilized loss, our quality-aware comparison function is simple and highly generalizable. The experiments conducted on several face recognition databases and benchmarks demonstrate that the introduced quality-awareness leads to consistent improvements in the recognition performance. Moreover, the proposed QMagFace approach performs especially well under challenging circumstances, such as crosspose, cross-age, or cross-quality. Consequently, it leads to state-of-the-art performances on several face recognition benchmarks, such as 98.50% on AgeDB, 83.97% on XQLFQ, and 98.74% on CFP-FP.




## Key Points

todo

To summarize, the proposed Pixel-Level Quality Assessment approach 
- can be applied on arbitrary FR networks,
- does not require training, 
-  and provides a pixel-level utility description of an input face explaining how well pixels in a face image are suited for recognition (prior to any matching).

The solution can explain why an image cannot be used as a reference image during the acquisition/enrolment process and in which area of the face the subject have to do changes to increase the quality. Consequently, PLQ maps provide guidance on the reasons behind low quality images, and thus can provide interpretable instructions to improve the FIQ.


## Results

The proposed approach is analysed in three steps. 
First, we report the performance of QMagFace on six face recognition benchmarks against ten recent state-of-the-art methods in image- and video-based recognition tasks to provide a comprehensive comparison with state-of-the-art. 
Second, we investigate the face recognition performance of QMagFace over a wide FMR range to show its suitability for a wide variety of applications and to demonstrate that the quality-aware comparison score constantly enhances the recognition performance. 
Third, we analyse the optimal quality weight over a wide threshold range to demonstrate the robustness of the training process and the generalizability of the proposed approach.

In the following, we will only show some results. For more details and dicussions, please take a look at the paper.



**PLQ explanation maps before and after inpainting** - Images before and after the inpainting process are shown with their
corresponding PLQ-maps and FIQ values. The images show the effect of small and large occlusions, glasses, headgears, and beards on the
PLQ-maps for two FR models. In general, these are identified as areas of low pixel-quality and inpainting these areas strongly increases
the pixel-qualities of these areas as well as the FIQ. This demonstrates that our solution leads to reasonable pixel-level quality estimates
and thus can give interpretable recommendations on the causes of low quality estimates.

<img src="Results ArcFace Inpainting.png"  > 

**PLQ-explanation maps for random masks** - For two random identities, their masked and unmasked images are shown with
their corresponding PLQ-maps. In general, the effect of the mask on the PLQ-map is clearly visible demonstrating the effectiveness of the
proposed approach to detect disturbances.

<img src="Results ArcFace Masking.png"  > 

**PLQ-explanation maps for ICAO imcompliant images** - One ICAO-compliant image and twelve images with imcompliances
are shown with their corresponding PLQ-maps. Occlusions (b,c,d), distorted areas of the face (f), and reflections result in low-pixel
qualities.

<img src="Results ArcFace ICAO.png"  > 


## Installation
to do Malte






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

