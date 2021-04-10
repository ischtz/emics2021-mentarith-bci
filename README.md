# emics2021-mentarith-bci

Data and code for our paper *"Towards Robust Binary Communication Using Pupil Dilation by Mental Arithmetic"*, accepted at the [EMICS Workshop at CHI 2021](https://emics-2021.github.io/EMICS/).

## Abstract 

The pupil reliably dilates with increases in cognitive load, e.g., when performing mental multiplication. This mechanism has previously
been used for communication with Locked-In-Syndrome patients (Stoll et al., 2013), providing a proof-of-principle without focusing on
communication speed. We investigated a similar paradigm in a larger sample of healthy participants (N=24) using three different eye
trackers, including a sub-$100 commercial solution. Using the previously suggested training-free approach, we reliably decoded binary
(yes/no) responses with comparable performance in all devices. To address whether throughput can be increased without sacrificing
accuracy, we explored different classification models. Using a support-vector machine (SVM), we reached useful decoding with as little
as 5 s of data. Our results confirm that mental arithmetic-based pupil dilation is a viable communication solution and suggests that
bitrates of high-end brain-computer interfaces (BCI) can be reached at substantially lower cost.

![figure2](https://user-images.githubusercontent.com/7711674/114265043-ddf4b480-99ee-11eb-8fdb-72bd03bdda2e.png)

**Paper Fig. 2.** Decoding Analysis. Left: AUC of the best-performing SVM model when trained and tested on time windows in steps of 250 ms
(7-fold CV). Middle: Average cross-validation performance (AUC) of different algorithms in trial decoding (left; circles) and response
interval decoding (right; squares). DS: Difference slope method - timing as in Stoll et al., 2013; DS/full: DS method applied to full trial duration.
Right: Best model (SVM) AUC when trained and tested on separate response intervals, 250 ms intervals. Error bars / shading: AUC
range in 7-fold CV. Black dashed line: chance level.

## Requirements

- Uses [pyedfread](https://github.com/nwilming/pyedfread) to import raw Eyelink EDF files


## Citation

If you reference or use any of the materials in this repository, please cite our workshop paper: 

*Immo Schuetz, Julia Trojanek, and Wolfgang Einhäuser. 2021. Towards Robust Binary Communication Using Pupil Dilation by Mental
Arithmetic. In EMICS ’21: ACM CHI ’21 Workshop on Eye Movements as an Interface to Cognitive State, May 14, 2021, Yokohama, Japan.
ACM, New York, NY, USA, 6 pages.*

