#python3 PiCameraStream.py -c yolov3-tiny.cfg -w yolov3-tiny_10000.weights -cl yolo.names

#import the necessary packages
from imutils.video import VideoStream
import imutils
import time
import cv2
import argparse
import numpy as np
#add argument
ap= argparse.ArgumentParser()

ap.add_argument('-c','--config',required=True,
                    help='path to yolo config file')
ap.add_argument('-w','--weights',required=True,
                    help='path to yolo trained weights')
ap.add_argument('-cl','--classes',required=True,
                    help='path to text file containing class names')
args=ap.parse_args()

#return output layer
def get_output_layers(net):
    layer_names=net.getLayerNames()
    output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return output_layers
#return rectangle bouding boxs and class names
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label= str(classes[class_id])
    color= COLORS[class_id]
    cv2.rectangle(img,(x,y),(x_plus_w,y_plus_h),color,2)
    cv2.putText(img,label,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)


# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")

vs = VideoStream(usePiCamera=True, resolution=(640, 480)).start()
time.sleep(2.0)

#read classes name
classes= None
with open(args.classes,'r') as f:
    classes= [line.strip() for line in f.readlines()]
COLORS= np.random.uniform(0, 255, size= (len(classes),3))
net= cv2.dnn.readNet(args.weights,args.config)

i=1
while True:
    # grab the frame from the video stream and resize it to have a
    # maximum width of 400 pixels
    frame = vs.read()
    image = imutils.resize(frame, width=320)
    i+=1
    if i%10==0:
        #resize and put in neural network predict
        Width= image.shape[1]
        Height= image.shape[0]
        scale= 0.00392
        blob= cv2.dnn.blobFromImage(image,scale,(416,416),(0,0,0),True,crop=False)
        net.setInput(blob)
        outs=net.forward(get_output_layers(net))

        #remove noise object in frame
        class_ids=[]
        confidences=[]
        boxes=[]
        conf_threshold= 0.2
        nms_threshold= 0.4
        for out in outs:
            for detection in out:
                scores= detection[5:]
                class_id= np.argmax(scores)
                confidence= scores[class_id]
                if( confidence >0.5):
                    center_x= int(detection[0]*Width)
                    center_y= int(detection[1]*Height)
                    w= int(detection[2]*Width)
                    h= int(detection[3]*Height)
                    x=center_x- w/2
                    y=center_y- h/2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x,y,w,h])
        indices= cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold, nms_threshold)
        for i in indices:
            i=i[0]
            box= boxes[i]
            x= box[0]
            y= box[1]
            w= box[2]
            h= box[3]
            draw_prediction(image, class_ids[i], round(x),round(y), round(x+w), round(y+h))
            print(str(classes[class_id]))
         # show the output frame
        cv2.imshow("object_detection", image)
        key = cv2.waitKey(1) & 0xFF
        

    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()