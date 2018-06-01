## Project Logbook
---
### Monday - October 16th, 2017 - Akhass + Joan
Data to be used from ESA website: SAR level 1Single Look Complex
Tasks to be done:
Have a look at the data format, is it a vector image, number of bits per pixel
Find out what will be the ground truth
What sort of scanning technique
Refer to SAR book (to be suggested by Gareth)

### Thursday - October 19th, 2017 - Akhass + Flavio
References to research papers will be provided. Have a look how SAR data is being used
ESA Sentinel 1A sensor being used by IBM. IBM has cut out data from those images (training and testing data)
Preprocessing for SR data: books, research papers, ESA technical guide online
Ground truth: Automatic Identification System (AIS) self-reported regularly by ships, however Flavio found there was a misalignment in timestamps
Initial project milestone: segment the image into landmass / water/ coastline / ships, then have a look at improving ship detection using AIS data
NDA being dealt by IBM legal and CUED Director of Research
Export Classification required to share any info, code, etc.
Do not look at source code in git repo, report any bugs you find
Tasks to be done:
Email Flavio Bluemix IDs of Akhass and Joan so he can set up git repo
Email Flavio the following for Export Classification: name, Nationality, Visa type
Get accustomed to using SNAP toolkit and its workflows
Download a dataset from ESA OpenHub that shows coastline
Follow SNAP tutorial on ship detection using Constant False Alarm Rate (CFAR)
Understand limitations of CFAR algorithm

### Monday - October 23rd, 2017 - Akhass + Joan
Updated Joan on prev discussion with Flavio
Explained queries that arose in prev meeting
Format of SAR level 1 SLC data: TIFF, approx dimensions 21530 x 13480, resolution 96dpi x 96dpi, bitdepth 32
Ground truth will be AIS
Interferometric Wide swath (IW). It acquires data with a 250 km swath at 5 m by 20 m spatial resolution (single look)
Tasks to be done:
See what CFAR does
Get data with ships and without ships
Use CFAR as baseline method, to compare with subsequent CNN models
Potentially circumvent timestamp misalignment between image and AIS data: Could use ship location from AIS to map trajectory and predict location in subsequent images

### Monday - October 30th, 2017 - Akhass + Joan + Flavio
Sentinel 1 SAR mission overview:
Twin near polar-orbiting satellites in same orbit, 180 degrees out of phase, named Sentinel 1A and Sentinel 1B
C-band transmission: 5.405 GHz ~ 5cm
4 imaging modes: Stripmap (SM), Interferometric Wide swath (IW), Extra-Wide swath (EW), Wave (WV).
 Dual Polarisation: H-sent, V- sent. Received modes VV, VH, HH, HV
Explained CFAR algorithm
Introduced RUS VM with pre-installed SNAP toolkit, applied for access, yet to be granted access by ESA
Overview of different options for ground truth: AIS, LRIT
Tasks to be done:
Use GRD data
Apply orbit correction and land masking
Run CFAR algorithm to detect ships (look up papers on CFAR parameter settting)
Export workflow in CSV from SNAP
Run batch mode on SNAP toolkit
Master creating workflows and modifying
Look at CNN and other models e.g. spiking neural net

### Monday - November 6th, 2017- Akhass + Joan
Akhass able to create workflows in SNAP toolkit, save as .xml file
Gave demo of object detection (using CFAR) on SNAP toolkit in RUS VM. Orbit correction + land masking applied before CFAR. Used GRD data off the coast of France
Land masking used to mask out land. Shuttle Radar Topography Mission (SRTM) is an international research effort that obtained digital elevation models on  56° S to 60° N. These digital elevation points are used to identify landmasses and mask them out before CFAR.
At the moment, we are not able to check accuracy these CFAR detections. Therefore, for next week:
Choose suitable subimage with few ships
Manually label those ships using AIS data
Run CFAR algorithm on that subimage and calculate accuracy of detections
Further work to be done:
Try morphological operators on subimage, see how the results compare to CFAR
Read up on how differences in VV and VH images can be exploited
Progress reviewed: Some project marks are allocated for mid-of-term project review, project seems to be on track
Michaelmas Project Presentation is scheduled for 3pm on Friday 17th November. Akhass will be giving a mock presentation at the next meeting on Monday 13th November.
