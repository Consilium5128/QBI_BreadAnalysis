## Quantifying Pore Distributions from Bread Samples

### Course Project for Quantitative Big Imaging: From Images to Statistics (ETH Zurich - Spring'24)
Under the kind Mentorship of **Dr. Anders Kaestner (PSI)**

We have 4 bread samples - Ro1a, Ru1a, V1a, W1a. Here we try to quantify the pore distributions in these samples acquired using X-ray tomography experiments.

This repository is still in it's infancy (born Yeast-erday) and uses .ipynb notebooks to define the workflow, while importing some functions to save memory (and time).

The required dependencies can be installed by creating a Conda environment using environment_bread.yml file

The basic preprocessing and processing pipeline is in the bread_workflow.ipynb including loading original images, downsampling, (anisotropy detection), filtering, thresholding, clustering, pore extraction, segmentation, regionprops, thickness and distance maps, and finally saving the images and metric DataFrames generated. The images are currently stored on the Euler cluster (scratch) of ETH Zurich.
(Test.ipynb shows the (half-baked) messy experimentation)

Some metric testing has been then carried out in bread_prelim_analysis.ipynb including further image analysis, metric analysis, clustering, and statistical tests.

#### To be updated

