import time
import edgeiq
from networktables import NetworkTables
from cscore import CameraServer
import logging
import numpy as np

# Constant for the default confidence (0 being 0% sure and 1 being 100% sure)
default_conf_thres = .25
default_width = 640
default_height = 320
max_framerate = 10
counter = 0

def main():

    global counter
    global max_framerate

    # Allow Rio to boot and configure network
    time.sleep(10.0)

    # Setup logging for the NetworkTables messages
    logging.basicConfig(level=logging.DEBUG)

    # Setup NetworkTables
    NetworkTables.initialize(server = '10.8.34.2')

    # Create table for values
    EVS = NetworkTables.getTable('EVS')
    sd = NetworkTables.getTable('SmartDashboard')

    # Set default run_vision_tracking value
    EVS.putBoolean('run_vision_tracking', True)
    EVS.putNumber('confidence_thres', default_conf_thres)
    EVS.putNumber('max_framerate', max_framerate)
    EVS.putNumber('stream_width', default_width)
    EVS.putNumber('stream_height', default_height)

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

    # Get the FPS
    fps = edgeiq.FPS()

    # Setup the tracker for 20 frames deregister time and have a matching tolerance of 50
    tracker = edgeiq.CentroidTracker(deregister_frames=20, max_distance=50)

    # Setup video cam feed
    cs = CameraServer.getInstance()
    cs.enableLogging()
    outputStream = cs.putVideo("Vision_Out", 300, 300)

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream: 
            
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            for i in range(0,9): 
                Power_CellTables[i].putBoolean('inUse', False)

            for i in range(0,0): 
                GoalTables[i].putBoolean('inUse', False)

            last_update_time = time.time()

            # loop detection
            while True:

                # Pull a frame from the camera
                frame = video_stream.read()

                # Check to see if the camera should be processing images
                if (EVS.getBoolean('run_vision_tracking', True)):
                    
                    # Process the frame
                    results = obj_detect.detect_objects(frame, confidence_level = EVS.getNumber('confidence_thres', default_conf_thres))

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
                    frame = edgeiq.markup_image(frame, results.predictions, colors=colors)

                    # Update the FPS tracker
                    fps.update()

                
                # Get the parameters for streaming
                max_framerate = EVS.getNumber('max_framerate', max_framerate)
                width = EVS.getNumber('stream_width', default_width)
                height = EVS.getNumber('stream_height', default_height)

                # Calculated the desired frame time
                desired_frame_time = 1 / max_framerate

                # Take the last update time from the current time. It gives the time since the last frame. 
                # If the time is larger, than we should stream, otherwise skip streaming the frame.
                if (time.time() - last_update_time >= desired_frame_time):
    
                    # Time to stream a frame!

                    # Resize to the desired size
                    frame = edgeiq.resize(frame, width, height)

                    # Put the frame out to the stream
                    outputStream.putFrame(frame)

                    # Update the last update time
                    last_update_time = time.time()

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")
if __name__ == "__main__":
    main()