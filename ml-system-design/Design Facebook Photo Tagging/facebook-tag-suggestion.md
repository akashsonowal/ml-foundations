# Design Facebook Photo Tagging

Design a feature that reduces the effort of tagging people in photos when a user uploads a photo to Facebook.

Q: When you say reduce the effort of tagging people in photos upon upload, do you mean automatically tagging or just presenting suggestions on which users should be tagged in the photo?

A: Let's present a list of suggestions.

Q: Will these suggestions be in a dropdown menu, perhaps appearing when a user clicks on a face?

A: Yes, there should be a box surrounding people's faces in photos such that when uploading the photo, users have the chance to click the boxes and tag a user.

Q: Do I need to design the ingestion of these photos into some analysis system?

A: Assume that the photos you have access to come into an HDFS cluster from a batch ingestion each day.

Q: Do I have to design the system which serves the tag suggestions, or just the model(s)?

A: Let's not focus on serving the predictions.

Q: Okay, do I need to design the ingestion of these photos into some analytics system?

A: Assume that the photos you have access to come into an HDFS cluster from a batch ingestion each day from our PostgreSQL cluster.

Q: Can I assume that these images are in HDF5 format under 30MB with dimensions under 1500x1500?

A: Yes, that's fine.

Q: Can I use pre-trained models?

A: Yes, but you have to explain how you'd train and use them, and how they work.

Q: Can I assume that we have access to a workforce which can label images?

A: Yes, but go into detail on how you'd use this workforce to label images.

Q: Can I assume that we only need to detect frontal faces, no occlusions, and no severe illumination problems?

A: To start out with, yes. At the end of the interview, we'll revisit this.

Q: Will we want to detect faces from various distances away from the camera lens?

A: Yes, you should design a system which support multiple scales of faces.

Q: I'm assuming that we want to be able to detect more than one face per image?

A: Yes. That's right.

## Solution Walkthrough

### 1. Gathering System Requirements

As with any ML design interview question, the first thing that we want to do is gather requirements. We need to figure out what we need to build and what we don't need to build.

We're designing a system which allows users to more easily tag other users in photos they upload by presenting suggestions for which users are in the photo.

To accomplish this, we'll need to do 5 ML-related tasks to implement tag suggestions:

Image and Label Collection
Image Pre-processing
Face Detection
Face Recognition
Performance Measurement
We'll also need to rely on systems which we won't touch on or design:

The website's UI displaying the suggestions
The inference service providing the suggestions to the UI
The HDFS cluster and underlying processing cluster(s)

### 2. Image and Label Collection

Assuming we have access to images, we'll need two roughly equal sized samples of labeled images. One sample where each image contains one or more faces and another sample where each image contains no faces. The images with faces in them should contain faces at different scales, contrasts, poses, and facial expressions. To start it should include frontal faces with no occlusions or illumination problems. We'll start with roughly 5000 samples of each image class, faces and no faces.

To get the images labeled we'll need some annotation software which provides images and to a workforce so that the images can be labeled appropriately. The annotation software should provide examples of what frontal face images are and are not. This will provide labels for the images so that we can train a model which detects faces in images.
