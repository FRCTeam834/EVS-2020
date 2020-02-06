import time
import edgeiq
import pyfrc
from networktables import NetworkTables
import logging
import numpy as np

# Constant for the default confidence (0 being 0% sure and 1 being 100% sure)
default_conf_thres = .75

def main():
    # Allow Rio to boot and configure network
    time.sleep(5.0)

    # Setup logging for the NetworkTables messages
    logging.basicConfig(level=logging.DEBUG)

    # Setup NetworkTables
    NetworkTables.initialize(server = '10.8.34.2')

    # Create table for values
    EVS = NetworkTables.getTable('EVS')
    sd = NetworkTables.getTable('SmartDashboard')

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
            "alwaysai/mobilenet_ssd")
    obj_detect.load(engine=edgeiq.Engine.DNN_OPENVINO)
 

    # Print out info
    print("Loaded model:\n{}\n".format(obj_detect.model_id))
    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Labels:\n{}\n".format(obj_detect.labels))

    # Get the FPS
    fps = edgeiq.FPS()
    tracker = edgeiq.CentroidTracker()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream: 
            
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            for i in range(0,9): 
                Power_CellTables[i].putBoolean('inUse', False)

            for i in range(0,0): 
                GoalTables[i].putBoolean('inUse', False)

            # loop detection
            while True:

                frame = video_stream.read()
                results = obj_detect.detect_objects(frame, confidence_level = default_conf_thres)
                tracked_objects = tracker.update(results)

                # Counters - they reset after every frame in the while loop 
                Power_CellCounter = 0 
                GoalCounter = 0 
        
                # Update the EVS NetworkTable with new values
                for prediction in tracked_objects:                                                                                                                        

                    center_x, center_y = prediction.box.center
                    # Code goes here
                    numValues = [center_x, center_y, prediction.box.end_x, prediction.box.end_y, prediction.box.area, (prediction.confidence * 100)]
                    
                    for entry in range(0, len(numValues) - 1):
                        numValues[entry] = round(numValues[entry], 3)

                    numValuesArray = np.asarray(numValues) 

                    if prediction.label == "Power_Cell":
     
                        Power_CellTables[Power_CellCounter].putNumberArray('values', numValuesArray)
                        # Boolean asks to update
                        Power_CellTables[Power_CellCounter].putBoolean('inUse', True)

                        Power_CellCounter += 1 

                    if prediction.label == "Goal":
     
                        GoalTables[GoalCounter].putNumberArray('values', numValuesArray)
                        # Boolean asks to update
                        GoalTables[GoalCounter].putBoolean('inUse', True)

                        GoalCounter += 1 

                # Sets the value after the last value to false. The Rio will stop when it finds a False
                Power_CellTables[Power_CellCounter].putBoolean('inUse', False)
                GoalTables[GoalCounter].putBoolean('inUse', False)

                EVS.putBoolean('checked', False)
                fps.update()
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")
if __name__ == "__main__":
    main()