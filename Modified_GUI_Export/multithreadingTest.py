import time
import random
import threading

times = []

max_framerate = 2

sleep_time = 0
printing_time = 0


def runOtherStuff():

    global sleep_time
    global max_framerate

    while True:
        start_time = time.time()
        sleep_time = random.random()
        end_time = time.time()
        time_taken = end_time - start_time
        time.sleep((1/max_framerate) - time_taken)

def putFrame():
    global sleep_time
    global printing_time
    #global max_framerate

    while True:
        #time.sleep(1/max_framerate)
        printing_time = sleep_time

def main():
    global printing_time
    runOtherStuffThread = threading.Thread(target = runOtherStuff, daemon=True)
    putFrameThread = threading.Thread(target = putFrame, daemon = True)

    runOtherStuffThread.start()
    putFrameThread.start()

    while True:
        print(printing_time)
        time.sleep(.1)

main()
