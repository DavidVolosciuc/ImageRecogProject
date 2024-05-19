#bibliography:
#https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606
#https://www.tutorialspoint.com/how-to-detect-humans-in-an-image-in-opencv-python
#https://imageai.readthedocs.io/en/latest/
#https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
#https://imageai.readthedocs.io/en/latest/detection/
# importing libraries
from imageai.Detection import ObjectDetection
import os

# execution path of script
execution_path = os.getcwd()

# initialize ImageAI library
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()

# Set the detection to only work on people
custom = detector.CustomObjects(person=True)
# Get all the images in the poze folder
for filename in os.listdir('poze'):
    f = os.path.join("poze", filename)
    if os.path.isfile(f):
        # run the detection for each image and save it to pozedupa folder
        detections = detector.detectObjectsFromImage(custom_objects=custom,input_image=os.path.join(execution_path,f), output_image_path= os.path.join(execution_path, "pozedupa/{}".format(filename)))
        # print each person with their probability
        for eachObject in detections:
            print(eachObject["name"] , " : " , eachObject["percentage_probability"] )