import time
import edgeiq
from networktables import NetworkTables
from cscore import CameraServer
import logging
import numpy as np
import threading
from collections import deque

# Set up variables
default_conf_thres = .25  # Decimal version of the percentage
max_framerate = 10
default_width = 640
default_height = 320
EVS = None
start_streaming = False

# Create the queues for the unmarked and marked images
unmarked_queue = deque(maxlen=2)
marked_queue = deque(maxlen=2)

# Setup NetworkTables
def visionProcessing():

    # Load in all of the universal variables
    global EVS
    global start_streaming

    # Create sub-tables and append them to arrays
    Power_CellTables = []
    Power_Cell0 = EVS.getSubTable('Power_Cell0')
    Power_CellTables.append(Power_Cell0)
    Power_Cell1 = EVS.getSubTable('Power_Cell1')
    Power_CellTables.append(Power_Cell1)
    Power_Cell2 = EVS.getSubTable('Power_Cell2')
    Power_CellTables.append(Power_Cell2)
    Power_Cell3 = EVS.getSubTable('Power_Cell3')
    Power_CellTables.append(Power_Cell3)
    Power_Cell4 = EVS.getSubTable('Power_Cell4')
    Power_CellTables.append(Power_Cell4)
    Power_Cell5 = EVS.getSubTable('Power_Cell5')
    Power_CellTables.append(Power_Cell5)
    Power_Cell6 = EVS.getSubTable('Power_Cell6')
    Power_CellTables.append(Power_Cell6)
    Power_Cell7 = EVS.getSubTable('Power_Cell7')
    Power_CellTables.append(Power_Cell7)
    Power_Cell8 = EVS.getSubTable('Power_Cell8')
    Power_CellTables.append(Power_Cell8)
    Power_Cell9 = EVS.getSubTable('Power_Cell9')
    Power_CellTables.append(Power_Cell9) 

    GoalTables = []
    Goal0 = EVS.getSubTable('Goal0')
    GoalTables.append(Goal0)

    # Setup EdgeIQ 
    obj_detect = edgeiq.ObjectDetection(
            "CAP1Sup/FRC_2020_834_v2") 
    obj_detect.load(engine=edgeiq.Engine.DNN_OPENVINO)

    # Setup color values for objects (in BGR format), and then combine them to a single scheme
    default_color = (0, 0, 255)
    color_map = {
        "Power_Cell": (0, 255, 255),
        "Goal": (255, 0, 0)
    }
    colors = [color_map.get(label, default_color) for label in obj_detect.labels]
 

    # Print out info
    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    # Setup the tracker for 20 frames deregister time and have a matching tolerance of 50
    tracker = edgeiq.CentroidTracker(deregister_frames=20, max_distance=50)

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream: 
            
            # Allow Webcam to warm up
            time.sleep(2.0)

            # Refresh the tables so that they are all false
            for i in range(0,9): 
                Power_CellTables[i].putBoolean('inUse', False)

            for i in range(0,0): 
                GoalTables[i].putBoolean('inUse', False)

            # Loop the detection
            while True:

                # Pull a frame from the camera
                unmarked_frame = video_stream.read()
                unmarked_queue.appendleft(unmarked_frame)

                # Check to see if the camera should be processing images
                if (EVS.getBoolean('run_vision_tracking', True)):
                    
                    # Process the frame
                    results = obj_detect.detect_objects(unmarked_frame, confidence_level = EVS.getNumber('confidence_thres', default_conf_thres))

                    # Update the object tracking
                    objects = tracker.update(results.predictions)

                    # Counters - they reset after every frame in the while loop 
                    Power_CellCounter = 0 
                    GoalCounter = 0 

                    # Define the collection variable
                    predictions = []
            
                    # Update the EVS NetworkTable with new values
                    for (object_id, prediction) in objects.items():

                        # Add current prediction to the total list
                        predictions.append(prediction)                                                                                                   

                        # Pull the x and y of the center
                        center_x, center_y = prediction.box.center

                        # Package all of the values as an array for export
                        numValues = [object_id, center_x, center_y, prediction.box.end_x, prediction.box.end_y, prediction.box.area, (prediction.confidence * 100)]
                        
                        # Round all of the values to the thousands place, as anything after is irrelevant to what we need to do
                        for entry in range(0, len(numValues) - 1):
                            numValues[entry] = round(numValues[entry], 3)

                        # Convert the values to a numpy array for exporting
                        numValuesArray = np.asarray(numValues) 

                        # Sort results into their respective classes
                        if prediction.label == "Power_Cell":
        
                            if (Power_CellCounter < 9):
                                Power_CellTables[Power_CellCounter].putNumberArray('values', numValuesArray)
                                # Boolean asks to update
                                Power_CellTables[Power_CellCounter].putBoolean('inUse', True)

                            Power_CellCounter += 1 

                        elif prediction.label == "Goal":
                            
                            if (GoalCounter < 1):
                                GoalTables[GoalCounter].putNumberArray('values', numValuesArray)
                                # Boolean asks to update
                                GoalTables[GoalCounter].putBoolean('inUse', True)

                            GoalCounter += 1 

                    # Sets the value after the last value to false. The Rio will stop when it finds a False
                    if (Power_CellCounter < 9):
                        Power_CellTables[Power_CellCounter].putBoolean('inUse', False)
                    
                    if (GoalCounter < 1):
                        GoalTables[GoalCounter].putBoolean('inUse', False)

                    # Notify the Rio that vision processing is done, and the data is valid again
                    EVS.putBoolean('checked', False)

                    # Do the frame labeling last, as it is lower priority
                    marked_frame = edgeiq.markup_image(unmarked_frame, results.predictions, colors=colors)
                    marked_queue.appendleft(marked_frame)
    
    finally:
        print('Program ending')


def stream():
    # Load all of the universal variables
    global max_framerate
    global EVS

    # Setup video cam feed
    cs = CameraServer.getInstance()
    outputStream = cs.putVideo("Vision_Out", 300, 300)

    while True:
        # Get the starting time
        start_time = time.time()

        # Get the latest streaming settings from NetworkTables
        max_framerate = EVS.getNumber('max_framerate', max_framerate)
        width = EVS.getNumber('stream_width', default_width)
        height = EVS.getNumber('stream_height', default_height)

        # Calculated the desired frame time
        desired_frame_time = 1 / max_framerate


        # Stream the images. Unmarked frames if vision isn't running, marked if it is
        if (EVS.getBoolean('run_vision_tracking', True)):
            try:
                marked_frame = marked_queue.pop()
                marked_frame = edgeiq.resize(marked_frame, width, height)
                outputStream.putFrame(marked_frame)
            except IndexError:
                # Queue is empty
                # Other options would be to skip the rest of the loop or wait
                # for an item to be added to the queue
                pass
        else:
            try:
                unmarked_frame = unmarked_queue.pop()
                unmarked_frame = edgeiq.resize(unmarked_frame, width, height)
                outputStream.putFrame(unmarked_frame)
            except IndexError:
                # Queue is empty
                pass

        # Get the ending time
        end_time = time.time()

        # Calculate the actual time taken for the frame
        actual_frame_time = end_time - start_time

        # If the frames are being processed too fast, we need to slow down
        if actual_frame_time < desired_frame_time:

            # Wait the time difference
            time.sleep(desired_frame_time - actual_frame_time)


def main():

    # Make a counter so it's easier to see if the processing is running
    counter = 0

    # Allow Rio to boot and configure network
    time.sleep(10.0)

    # Setup logging for the NetworkTables messages
    logging.basicConfig(level=logging.DEBUG)

    # Set up main EVS table
    NetworkTables.initialize(server = '10.8.34.2')

    # Create table for values
    EVS = NetworkTables.getTable('EVS')

    # Set up the NetworkTables values
    EVS.putBoolean('run_vision_tracking', True)
    EVS.putNumber('confidence_thres', default_conf_thres)
    EVS.putNumber('max_framerate', max_framerate)
    EVS.putNumber('stream_width', default_width)
    EVS.putNumber('stream_height', default_height)

    # Create the processing frames
    visionProcessingThread = threading.Thread(target = visionProcessing, daemon = True)
    streamingThread = threading.Thread(target = stream, daemon = True)

    # Start both of the processing frames
    visionProcessingThread.start()
    streamingThread.start()

    # Prevent the program from exiting so that the threads can run. Also, keep printing to show the program is running
    while True:
        print("Still running. Counter: " + str(counter))
        counter =+ 1

if __name__ == "__main__":
    main()
