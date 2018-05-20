import multiprocessing

from multiprocessing import Queue

from OpenPersonDetector import OpenPersonDetector
from newgen.GenerativeDetector import AbstractDetectorGenerator


class ManagedOPDetector:
    def __init__(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue

    def detectPersons(self, image, discardedGrayImage):
        self.input_queue.put(image)
        return self.output_queue.get()


class ManagedOPDetectorGenerator(AbstractDetectorGenerator):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue

    def generate_detector(self):
        return ManagedOPDetector(self.input_queue, self.output_queue)


class ManagedOP:
    def __init__(self):
        self.queue_pairs = []
        self.worker = None

    def obtainGenerator(self):
        input_queue = Queue()
        output_queue = Queue()
        self.queue_pairs.append((input_queue, output_queue))
        return ManagedOPDetectorGenerator(input_queue, output_queue)

    def _async_worker(self, queue_pairs):
        person_detector = OpenPersonDetector(preview=False)
        while True:
            for input_queue, output_queue in queue_pairs:
                if input_queue.qsize() > 0:
                    frame = input_queue.get()
                    person_detections = person_detector.detectPersons(frame, None)
                    output_queue.put(person_detections)

    def startAsync(self):
        self.worker = multiprocessing.Process(target=self._async_worker, args=(self.queue_pairs,))
        self.worker.daemon = True
        self.worker.start()
