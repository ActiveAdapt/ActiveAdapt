###✨Continual Environmental Adaptation in the Wild under Real-time Data Acquisition Guidance✨###

Contents:

1. Installation 
2. Datasets 
3. Detailed setting for each round 
4. Test set collection details 
5. Usage

###########################################################################################

1. Installation 

           -- Python 3.7
 
           -- pytorch 1.0.1
           
Please refer to https://github.com/jwyang/faster-rcnn.pytorch to get detailed prerequistes.

###########################################################################################

2. Datasets 

Our collected lemur dataset and constructed indoor dataset are shown in the "/data" directory.

![1670731845(1)](https://user-images.githubusercontent.com/119776995/206885991-81291667-94c8-415d-874f-65b6e07d5493.png)
![1670731877(1)](https://user-images.githubusercontent.com/119776995/206886003-1279678e-3b1f-492c-b502-364609c0f1f1.png)


###########################################################################################

3. Detailed setting for each round 

![1670731691(1)](https://user-images.githubusercontent.com/119776995/206885901-3373c002-241b-41c2-a67a-a556d591859c.png)

To simulate the realistic class-bias CL data streams, during each round, the user takes images of objects from two classes. 20 images are taken per round, with different numbers of images for each class. Rounds differ, with different scenes and different object poses

###########################################################################################

4. Test set collection details 

We pre-collect two diverse datasets containing images with different object poses and backgrounds for system evaluation, since it is inefficient and time-consuming for users in the wild to collect and label the test set in each round required in the CL evaluation. To have similar image numbers in two test sets, we collect and label 120 images for each class (480 in total) to construct the test dataset for the lemur scenario, and 50 images for each class (450 in total) to form the test dataset for the indoor object scenario.

###########################################################################################

5. Usage

Run the detection assistance on the server:

           uvicorn app_guidance:app --reload --host "Your IP address" --port "Your port"

The AR app is build by the Unity engine.
