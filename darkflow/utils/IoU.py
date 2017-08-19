
from collections import namedtuple
import numpy as np
import cv2
from darkflow.utils.pascal_voc_clean_xml import pascal_voc_clean_xml

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	#print "max of" ,boxA[0], boxB[0] ,"is",xA
	#print "max of", boxA[1], boxB[1] ,"is",yA
	#print "min of", boxA[2], boxB[2] ,"is",xB
	#print "min of" ,boxA[3], boxB[3] ,"is",yB
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	dx = (xB-xA+1)
	dy = (yB - yA + 1)
	if (dx>=0) and (dy>=0):
		interArea=dx*dy
	else:
		interArea=0
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea =(boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	totalArea=float(boxAArea + boxBArea - interArea)
 	#print interArea , totalArea
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / totalArea

	# return the intersection over union value
	return iou


def find_accuracy(self,detected,groundtruth,path):

	labels=[]
	with open(self.FLAGS.labels) as my_file:
		labels = my_file.readlines()
	dumps=pascal_voc_clean_xml(path,labels,parseOne=path)


    #print dumps[0][0],dumps[0][1][2]

	groundtruth=dumps[0][1][2]
	false_positives=0

	undetected=[]
	False_Positive=[]
	True_Positive=[]

	#print(detected)
	#print(groundtruth)
	for i in range(0, len(groundtruth)):
		#print groundtruth[i][1:]
		#print "iter",i
		max=0.0; match=-1
		for j in range(0, len(detected)):

			if(groundtruth[i][0]==detected[j][0]):
				IoU=bb_intersection_over_union(groundtruth[i][1:],detected[j][1:])
			#print IoU
				if IoU>max:
					max=IoU
					match=j
		if max<0.4:
			undetected.append(groundtruth[i])
		elif max>=0.4:
			True_Positive.append((groundtruth[i],detected[match]))


	#for a in True_Positive:
	#print a,"  ",bb_intersection_over_union(a[0],a[1])

	for box in detected:
		flag=0
		for pair in True_Positive:
			if pair[1]==box:
				flag=1
				#print "found box in the detected", box
		if flag==0:
			False_Positive.append(box)

	self.TP = getattr(self, 'TP', 0) + len(True_Positive)
	self.FP = getattr(self, 'FP', 0) +len(False_Positive)
	self.UN = getattr(self, 'UN', 0) +len(undetected)

	print ("undetected:", len(undetected), "    True positives:", len(True_Positive), "   False positives:", len(False_Positive))


	print(	self.TP ,self.FP ,	self.UN )
	accuracy=100*self.TP/(self.TP+self.FP+self.UN)
	print("Cum ACCURACY: ",accuracy)

    #detected not in undetected and tuples are false positives
   # check_accuracy( [[39, 63, 203, 112],[49, 75, 203, 125],[31, 69, 201, 125],[50, 72, 197, 121],[35, 51, 196, 110] ],
   # [ [54, 66, 198, 114],[42, 78, 186, 126],[44,55,175,146],[18, 63, 235, 135],[54, 72, 1998, 120]]
   # ,"d")

'''

    for detection in examples:
	    # load the image
	    image = cv2.imread(detection.image_path)

	    # draw the ground-truth bounding box along with the predicted
	    # bounding box
	    cv2.rectangle(image, tuple(detection.gt[:2]),
		    tuple(detection.gt[2:]), (0, 255, 0), 2)
	    cv2.rectangle(image, tuple(detection.pred[:2]),
		    tuple(detection.pred[2:]), (0, 0, 255), 2)

	    # compute the intersection over union and display it
	    iou = bb_intersection_over_union(detection.gt, detection.pred)
	    cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
		    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	    print("{}: {:.4f}".format(detection.image_path, iou))

	    # show the output image
	    cv2.imshow("Image", image)
	    cv2.waitKey(0)'''
