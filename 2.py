import cv2

# Load the image
img = cv2.imread('images/employee.png')

classnames= []
classfiles =  'files/thing.names'


with open(classfiles,'rt' ) as f:
    classnames= f.read().rstrip('\n').split('\n')
   # print(classnames)

    p= 'files/frozen_inference_graph.pb'
    v='files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

    net=cv2.dnn_DetectionModel(p,v)   # consulter et decouvrir le fichier p et v
    net.setInputSize(320,230)   #height and widget
    net.setInputScale(1.0/127.5) #mesure
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)    #config couleur

    classIds,confs,bbox = net.detect(img, confThreshold=0.5) #confThreshold=0.5  دقت صور 
    #print(classIds,bbox)
    for  classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=3)
        cv2.putText(img,classnames[classId-1],
                        (box[0]+10,box[1]+20),
                        cv2.FONT_H

                        )
        



# Display the image in a window
cv2.imshow('rakwan', img)

# Wait for a key press and close the window
cv2.waitKey(0)

