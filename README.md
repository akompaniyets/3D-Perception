# 3-D Perception #
#### Robotics Software Engineering ####
#### Term 1 ####
#### Project 3 ####

_Concepts/Skills Learned:_
  * RGB-D camera basics: PointClouds, OpenCV calibration, ROS image_pipeline
  * PointCloud filtering: voxel grid, pass-through, RANSAC, outlier removal
  * K-means clustering
  * DBSCAN
  * Support Vector Machine (SVM): machine learning basics

---

_Project Description:_

This particular project is based partially on the 2017 Amazon Robotics Challenge, in which roboticists were tasked with visually differentiating potential objects of interest and correctly picking each object up, afterward placing it in a designated bin. For visualization of this project, Gazebo and Rviz were once again implemented, with ROS as the functional interface.

In order to accomplish the desired tasks, a basic understanding of RGB-D cameras was provided, specifically in using OpenCV for proper perspective calibration and the eventual creation of PointCloud data. Then, a process was described for the manipulation of PointClouds in deriving clusters of desired objects -- this involved voxel grid downsizing, pass-through filtering, RANSAC algorithm use (for eliminating PointCloud data associated with the table objects are situated on), as well as outlier removal due to camera noise.

Continuing the concepts of PointCloud interpretation, both K-means clustering and DBSCAN were covered in the task of clustering PointClouds (DBSCAN chosen for the project, due to the number of potential clusters varying) into distinct objects. Lastly, with a PointCloud isolated for each object of interest, machine learning was implemented with the SVM algorigthm, using supplied HSV and Normal Vector histograms for known objects in the development of a trained model. This model was then applied to the objects of interest to provide correct labels.

With Project 2 (Robot Arm Pick & Place) focusing heavily on the kinematics of robot arms, the action step of this project was simplified with a supplied movement function, which handled basic kinematics when provided with the centroid locations of identified objects.
     
   For a full report of how this was accomplished, see the included write-up: 
   
   [3-D Perception Write-Up](https://github.com/akompaniyets/3D-Perception/blob/master/3D%20Perception%20Pick%20and%20Place%20Write-Up.pdf)
