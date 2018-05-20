import multiprocessing
from abc import abstractmethod
from threading import Thread

import cv2
import numpy as np

from newgen.GenerativeDetector import AbstractInputFeeder, GenerativeDetector, AbstractFrameProcessor
from newgen.OPManager import ManagedOP

"""
Handle multiple camera inputs accross multiple processors. 
"""


class DetectionSplitter(AbstractFrameProcessor):
    '''
    Result of MultiCamProcessor.registerOPStream. Used to attach processors to OpenPose results stream.
    '''

    def __init__(self):
        super().__init__()
        self.processors = []

    def init(self):
        for processor in self.processors:
            if hasattr(processor, "init"):
                processor.init()

    def process_frame(self, processed_frame):
        '''
        Called by main process, providing detected/generated frame to user
        :param processed_frame: Detected or generated Frame
        :return:
        '''

        to_delete_processors = []

        for processor in self.processors:
            if not processed_frame.is_detected:  # Generated frame
                if hasattr(processor, "on_generated_detections"):
                    r = processor.on_generated_detections(processed_frame.raw_frame, processed_frame.detections,
                                                          processed_frame.time_frame)
                    if not r: to_delete_processors.append(processor)
            else:
                if hasattr(processor, "on_detections"):
                    r = processor.on_detections(processed_frame.raw_frame, processed_frame.detections,
                                                processed_frame.time_frame)
                    if not r: to_delete_processors.append(processor)

        for processor in to_delete_processors:
            print("Removing processor {}".format(processor))
            self.processors.remove(processor)

        return len(self.processors) > 0


class VideoCaptureInputFeeder(AbstractInputFeeder):

    def __init__(self, cap, scale=(0.5, 0.5)):
        super().__init__()
        self.cap = cap
        self.total_input_frames = 0
        self.scale = scale

    def init(self):
        pass

    def feed_input(self):
        # TODO: Add flow rate logic here

        r, frame = self.cap.read()
        if r:
            frame = np.array(frame, copy=True)
            frame = cv2.resize(frame, (0, 0), fx=self.scale[0], fy=self.scale[1])
            time_frame = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            # time_frame = cap.get_time()
            self.total_input_frames += 1
            return r, frame, time_frame
        else:
            return False, None, None


class _GenerativeDetectorSchedule:
    '''
    Internal
    '''

    def __init__(self, input_feeder, frame_processor, detector_generator):
        self.input_feeder = input_feeder
        self.frame_processor = frame_processor
        self.detector_generator = detector_generator


class MultiCamProcessor:
    '''
    Handle multiple cameras accross multiple processors
    '''

    def __init__(self):
        self.op_manager = ManagedOP()
        self.generative_detector_schedules = []
        self.worker_threads = []

    def registerOPStream(self, input_feeder):
        '''
        Registers an input frame feeder to OpenPose Detector and obtain a DetectionSplitter,
        which can be used to register detection processors.
        :param input_feeder:
        :return:
        '''
        detector_gen = self.op_manager.obtainGenerator()
        detection_splitter = DetectionSplitter()
        generative_detector_schedule = _GenerativeDetectorSchedule(input_feeder, detection_splitter, detector_gen)
        self.generative_detector_schedules.append(generative_detector_schedule)
        return detection_splitter

    def startSync(self):
        self.op_manager.startAsync()
        for generative_detector_schedule in self.generative_detector_schedules:
            worker_thread = Thread(target=GenerativeDetector().start_sync, args=(
                generative_detector_schedule.input_feeder, generative_detector_schedule.frame_processor,
                generative_detector_schedule.detector_generator))
            # TODO: Should this be daemon?
            worker_thread.start()
            self.worker_threads.append(worker_thread)

        for worker_thread in self.worker_threads:
            # Wait for all worker threads to complete
            worker_thread.join()


class AbstractProcessor:
    def __init__(self):
        super().__init__()

    @abstractmethod
    def init(self):
        '''
        Invoked once at startSync call of MulticamProcessor
        :return: void
        '''
        pass

    @abstractmethod
    def on_detections(self, raw_frame, detections, time_frame):
        '''
        Called for frames from detector
        :param raw_frame:
        :param detections:
        :param time_frame:
        :return: False to remove processor. True to keep processor alive.
        '''
        pass

    @abstractmethod
    def on_generated_detections(self, raw_frame, detections, time_frame):
        '''
        Called for frames generated by tracking
        :param raw_frame:
        :param detections:
        :param time_frame:
        :return: False to remove processor. True to keep processor alive.
        '''
        pass


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')


    class TestProcessor(AbstractProcessor):
        def __init__(self, title):
            self.title = title

        def on_detections(self, raw_frame, detections, time_frame):
            print("OnDetections", time_frame)
            cv2.imshow(self.title, raw_frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord("q"):
                return False
            return True

        def on_generated_detections(self, raw_frame, detections, time_frame):
            print("OnGeneratedDetections", time_frame)
            cv2.imshow(self.title, raw_frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord("q"):
                return False
            return True


    # Test
    multicam_processor = MultiCamProcessor()
    ips = VideoCaptureInputFeeder(cv2.VideoCapture("test_videos/ntb/head_office/Cash_Counter_1-1.dav"))
    splitter = multicam_processor.registerOPStream(ips)

    splitter.processors.append(TestProcessor("TestProcessor1"))
    splitter.processors.append(TestProcessor("TestProcessor2"))
    multicam_processor.startSync()
