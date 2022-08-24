# USAGE
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/massachusetts.mp4 --output output/massachusetts_output.avi
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/toronto.mp4 --output output/toronto_output.avi

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to deep learning segmentation model")
ap.add_argument("-c", "--classes", required=True,
	help="path to .txt file containing class labels")
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-s", "--show", type=int, default=1,
	help="whether or not to display frame to screen")
ap.add_argument("-l", "--colors", type=str,
	help="path to .txt file containing colors for labels")
ap.add_argument("-w", "--width", type=int, default=400,
	help="desired width (in pixels) of input image")
args = vars(ap.parse_args())



CLASSESO = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORSO = np.random.uniform(0, 255, size=(len(CLASSESO), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


# load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

# if a colors file was supplied, load it from disk
if args["colors"]:
	COLORS = open(args["colors"]).read().strip().split("\n")
	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	COLORS = np.array(COLORS, dtype="uint8")

# otherwise, we need to randomly generate RGB colors for each class
# label
else:
	# initialize a list of colors to represent each class label in
	# the mask (starting with 'black' for the background/unlabeled
	# regions)
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
		dtype="uint8")
	COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
# initialize the legend visualization
legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")

# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
	# draw the class name + color on the legend
	color = [int(c) for c in color]
	cv2.putText(legend, className, (5, (i * 25) + 17),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
		tuple(color), -1)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["model"])

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["video"])
writer = None

# try to determine the total number of frames in the video file
try:
	prop =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# construct a blob from the frame and perform a forward pass
	# using the segmentation model
	frame = imutils.resize(frame, width=args["width"])
	(h, w) = frame.shape[:2]
	
	#blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5,swapRB=True, crop=False)
    #net.setInput(bloba)
    
    #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (1024, 800)), 
    #	0.007843, (1024, 1024), 127.5)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (512, 512), 0,swapRB=True, crop=False)
	net.setInput(blob)
    
	detections = net.forward()
	
	#bloba = cv2.dnn.blobFromImage(frame, 1 / 255.0, (512, 256), 0,
	#	swapRB=True, crop=False)
	#net.setInput(bloba)
	start = time.time()
	output = net.forward()
	end = time.time()
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.2:
            # extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSESO[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORSO[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORSO[idx], 2)
			cv2.imshow("obj", frame)

		fps.update()

	# infer the total number of classes along with the spatial
	# dimensions of the mask image via the shape of the output array
	(numClasses, height, width) = output.shape[1:4]

	# our output class ID map will be num_classes x height x width in
	# size, so we take the argmax to find the class label with the
	# largest probability for each and every (x, y)-coordinate in the
	# image
	classMap = np.argmax(output[0], axis=0)

	# given the class ID map, we can map each of the class IDs to its
	# corresponding color
	mask = COLORS[classMap]

	# resize the mask such that its dimensions match the original size
	# of the input frame
	mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
		interpolation=cv2.INTER_NEAREST)

	# perform a weighted combination of the input frame with the mask
	# to form an output visualization
	output = ((0.2 * frame) + (0.8 * mask)).astype("uint8")

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(output.shape[1], output.shape[0]), True)
		print("ok")
		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(output)

	# check to see if we should display the output frame to our screen
	if args["show"] > 0:
		cv2.imshow("Frame", output)
		#cv2.imshow("real", frame)
		cv2.imshow("Legend", legend)
		key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# release the file pointers
print("[INFO] cleaning up...")
#writer.release()
vs.release()
