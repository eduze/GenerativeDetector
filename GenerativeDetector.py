import cv2
import numpy as np
import time
from abc import ABC, abstractmethod
from threading import Thread

from cv2 import VideoCapture
from multiprocessing import Queue, Process
from multiprocessing import Value

from OpenPersonDetector import OpenPersonDetector
from newgen.TrackJoin import TrackJoin


class PersonDetection:
    '''
    Class used to represent persons in generated frames
    '''

    def __init__(self, person_bound):
        self.person_bound = person_bound
        self.central_point = (
            int((person_bound[0] + person_bound[2]) / 2), int((person_bound[1] + person_bound[3]) / 2))
        self.track_index = None
        self.short_track_index = None
        self.head = None


class Frame:
    '''
    Processed frame provided to frame processor (may be generated or detected)
    '''

    def __init__(self, raw_frame, time_frame, detections=None):
        self.raw_frame = raw_frame  # Numpy array of raw pixels
        self.time_frame = time_frame
        self.detections = detections  # Person detections in frame
        self.is_detected = False


class _FrameBulk:
    def __init__(self, head_frame, tail_frames):
        self.head_frame = head_frame
        self.tail_frames = tail_frames


class AbstractInputFeeder(ABC):
    '''
    Base class for input video frames feeder
    '''

    def __init__(self):
        super().__init__()

    @abstractmethod
    def init(self):
        '''
        Invoked at the initiation of input process
        :return: void
        '''
        pass

    @abstractmethod
    def feed_input(self):
        '''
        Called by input process for requesting a video frame
        :return: (boolean : frame_present, 2D numpy array: frame, float: time_frame)
        '''
        pass


class AbstractFrameProcessor(ABC):
    '''
    Abstract class for processing output frames
    '''

    def __init__(self):
        super().__init__()

    @abstractmethod
    def init(self):
        '''
        Called by main process, during the call to start_sync, following initiation of system.
        :return:
        '''
        pass

    @abstractmethod
    def process_frame(self, processed_frame):
        '''
        Called by main process, providing detected/generated frame to user
        :param processed_frame: Detected or generated Frame
        :return:
        '''
        pass


class AbstractDetectorGenerator(ABC):
    '''
    Generates and provides a person detector
    '''

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate_detector(self):
        '''
        Invoked at the initiation of person detection thread.
        :return: Instance of person detector to be used
        '''
        pass


class GenerativeDetector:
    '''
    Analyses a realtime video frame stream using the provided person detector.
    The detections in missed frames by detector (due to time taken for processing a single frame for detection) are
    generated using trackers.
    '''

    def __init__(self):
        self.schedule_queue = Queue()
        self.detector_results_queue = Queue()
        self.all_frames_queue = Queue()

        mul = 3
        self.schedule_queue_capacity = mul * 3
        self.detector_results_queue_capacity = mul * 5
        self.results_queue_capacity = mul * 3
        self.max_hidden_frame_count = mul * 5

        self.detection_frame_time = Value("d", 0.0)
        self.tracking_frame_time = Value("d", 0.0)
        self.hidden_frame_count = Value("i", 0)

        self.frame_generator_process = None
        self.detector_process = None

    def _input_feed_thread(self, input_feeder):
        input_feeder.init()
        while True:
            while self.schedule_queue.qsize() < self.schedule_queue_capacity and self.detector_results_queue.qsize() < self.detector_results_queue_capacity and self.hidden_frame_count.value < self.max_hidden_frame_count:
                r, frame, time_frame = input_feeder.feed_input()
                if r:
                    self.schedule(frame, time_frame)

    def _frame_generator_thread(self, _detector_results_queue, _all_results_queue, tracking_frame_time,
                                hidden_frame_count):
        track_join = TrackJoin()
        while True:
            frame_bulk = _detector_results_queue.get()

            track_join.process_bulk(frame_bulk)

            head_frame = frame_bulk.head_frame
            hidden_frame_count.value -= 1

            _all_results_queue.put(head_frame)

            trackers = []
            track_indices = []
            head_links = []

            for detection in head_frame.detections:
                detection.head = detection
                tracker = cv2.TrackerMedianFlow_create()
                # tracker = cv2.TrackerKCF_create()
                person_bound = tuple(map(int, detection.person_bound))
                person_bound = (
                    person_bound[0], person_bound[1], person_bound[2] - person_bound[0],
                    person_bound[3] - person_bound[1])
                ok = tracker.init(head_frame.raw_frame, person_bound)
                if ok:
                    trackers.append(tracker)
                    track_indices.append(detection.track_index)
                    head_links.append(detection)

            for tail_frame in frame_bulk.tail_frames:
                tail_frame.detections = []
                tracking_start_time = time.time()
                for _i, tracker in enumerate(trackers):
                    ok, bbox = tracker.update(tail_frame.raw_frame)
                    if ok:
                        gen_person_bound = (int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        generated_detection = PersonDetection(gen_person_bound)
                        generated_detection.track_index = track_indices[_i]
                        generated_detection.head = head_links[_i]
                        tail_frame.detections.append(generated_detection)
                    # else:
                    #     print("Track Lost", track_indices[_i])
                tracking_end_time = time.time()
                tracking_frame_time.value = tracking_frame_time.value * 0.5 + (
                        tracking_end_time - tracking_start_time) * 0.5

                hidden_frame_count.value -= 1
                _all_results_queue.put(tail_frame)

    def print_stats(self):
        '''
        Print buffer status to standard output
        :return:
        '''
        print("Input Queue Size:", self.schedule_queue.qsize())
        print("Detection Results Queue Size:", self.detector_results_queue.qsize())
        print("Final Results Queue Size:", self.all_frames_queue.qsize())

    def _detector_thread(self, _schedule_queue, _results_queue, detector_generator, detection_frame_time,
                         hidden_frame_count):
        from trinet_reid.trinet import TriNetReID

        detector = detector_generator.generate_detector()

        re_id = TriNetReID()

        while True:
            frame = _schedule_queue.get()
            hidden_frame_count.value += 1

            detection_start_time = time.time()
            detections = detector.detectPersons(frame.raw_frame, None)

            person_crops = []
            _detections = []
            for __i, detection in enumerate(detections):
                minx, miny, maxx, maxy = detection.upper_body_bound
                person_crop = frame.raw_frame[
                              int(max(0, miny - (maxy - miny) / 1.8 - 5)):int(min(frame.raw_frame.shape[0], maxy + 5)),
                              int(max(0, minx - 5)):int(min(frame.raw_frame.shape[1], maxx + 5))]

                if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                    # detection.re_id_encoding = re_id.embed(person_crop)
                    person_crops.append(person_crop)
                    _detections.append(detection)
            # cv2.waitKey(1)
            embeddings = re_id.embed(person_crops)
            for i in range(len(_detections)):
                _detections[i].re_id_encoding = embeddings[i]

            detection_end_time = time.time()

            detection_frame_time.value = detection_frame_time.value * 0.5 + (
                    detection_end_time - detection_start_time) * 0.5

            frame.detections = detections
            frame.is_detected = True

            backlog = []
            while _schedule_queue.qsize() > 0:
                backlog.append(_schedule_queue.get())
                hidden_frame_count.value += 1

            frame_bulk = _FrameBulk(frame, backlog)

            _results_queue.put(frame_bulk)

    def schedule(self, frame, time_frame):
        '''
        Manually schedule a frame to input queue.
        :param frame: numpy array
        :param time_frame: time in seconds
        :return:
        '''
        self.schedule_queue.put(Frame(frame, time_frame))

    def has_results(self):
        '''
        Check whether any processed frames are present in the output buffer.
        :return:
        '''
        return self.all_frames_queue.qsize() > 0

    def get_result(self):
        '''
        Obtain a frame from output buffer
        :return:
        '''
        return self.all_frames_queue.get()

    def start_sync(self, input_feeder, frame_processor, detector_generator):
        '''
        Start processing input to generate detections on frames
        :param input_feeder:
        :param frame_processor:
        :param detector_generator:
        :return:
        '''
        self.frame_generator_process = Process(target=self._frame_generator_thread, args=(
            self.detector_results_queue, self.all_frames_queue, self.tracking_frame_time, self.hidden_frame_count))
        self.detector_process = Process(target=self._detector_thread, args=(
            self.schedule_queue, self.detector_results_queue, detector_generator, self.detection_frame_time,
            self.hidden_frame_count))
        self.frame_generator_process.daemon = True
        self.detector_process.daemon = True
        self.frame_generator_process.start()
        self.detector_process.start()

        self.input_feed_thread = Thread(target=self._input_feed_thread, args=(input_feeder,))
        self.input_feed_thread.daemon = True
        self.input_feed_thread.start()

        frame_processor.init()
        while True:
            processed_frame = self.get_result()
            r = frame_processor.process_frame(processed_frame)

            if not r:
                return


if __name__ == "__main__":
    class DetectorGenerator(AbstractDetectorGenerator):
        def generate_detector(self):
            return OpenPersonDetector(preview=False)


    class FrameProcessor(AbstractFrameProcessor):
        def init(self):
            self.colour_set = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                               (255, 125, 125), (125, 255, 125), (125, 125, 255), (255, 255, 125),
                               (255, 125, 255), (255, 255, 125)]

            cv2.namedWindow("preview", cv2.WINDOW_FREERATIO)

            self.start_time_map = {}

        def process_frame(self, processed_frame):
            frame = processed_frame.raw_frame
            time_frame = processed_frame.time_frame
            detections = processed_frame.detections

            for i, detection in enumerate(detections):
                colour = self.colour_set[detection.track_index % len(self.colour_set)]
                if detection.track_index not in self.start_time_map:
                    self.start_time_map[detection.track_index] = time_frame

                cv2.putText(frame, str(detection.track_index),
                            (int(detection.person_bound[0]), int(detection.person_bound[1])), cv2.FONT_HERSHEY_COMPLEX,
                            1, colour)

                time_elapsed = time_frame - self.start_time_map[detection.track_index]
                cv2.putText(frame, str(int(time_elapsed)) + " sec",
                            (int(detection.person_bound[0]), int(detection.person_bound[1] + 20)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, colour)

                cv2.rectangle(frame, (int(detection.person_bound[0]), int(detection.person_bound[1])),
                              (int(detection.person_bound[2]), int(detection.person_bound[3])), colour)

            cv2.imshow("preview", frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord("q"):
                return False

            return True


    class InputFeeder(AbstractInputFeeder):
        def init(self):
            self.cap = VideoCapture("test_videos/ntb/head_office/Cash_Counter_1-1.dav")
            self.total_input_frames = 0

        def feed_input(self):
            # TODO: Add flow rate logic here

            r, frame = self.cap.read()
            if r:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                frame = np.array(frame, copy=True)
                time_frame = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                # time_frame = cap.get_time()
                self.total_input_frames += 1
                return r, frame, time_frame
            else:
                return False, None, None


    generative_detector = GenerativeDetector()
    generative_detector.start_sync(input_feeder=InputFeeder(), frame_processor=FrameProcessor(),
                                   detector_generator=DetectorGenerator())
