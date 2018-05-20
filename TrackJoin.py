from newgen.utils import intersect_over_union_area, wall_movement, assumed_feet_movement
from trinet_reid.utils import compare


class TrackJoin:
    '''
    Conenct track segments by matching detections in batch head with previous batche's tail detections
    '''

    def __init__(self):
        self.last_bulk = None
        self.threshold_overlap = 0.01
        self.track_index_counter = -1
        self.re_id_threshold = 40

    def next_track_label(self):
        self.track_index_counter += 1
        return self.track_index_counter

    def _match(self, last_head, tail, head):
        # match tail detections to head (copy track indices from tail to head)
        tuples = []

        for tail_detection in tail.detections:
            for head_detection in head.detections:
                overlap_ratio = intersect_over_union_area(tail_detection.person_bound, head_detection.person_bound)
                wall_movement_ratio = 1 / wall_movement(tail_detection.person_bound, head_detection.person_bound)

                assumed_feet_movement_ratio = assumed_feet_movement(tail_detection.person_bound,
                                                                    head_detection.person_bound)

                if overlap_ratio > self.threshold_overlap:
                    head1 = tail_detection.head
                    head2 = head_detection
                    if head1.re_id_encoding is not None and head2.re_id_encoding is not None:
                        re_id_distance = compare(head1.re_id_encoding, head2.re_id_encoding)

                        if re_id_distance < self.re_id_threshold:
                            tuples.append((re_id_distance, tail_detection, head_detection))

        tuples = sorted(tuples, key=lambda a: a[0], reverse=False)
        mapped_head = set()
        mapped_tail = set()

        for score, tail_detection, head_detection in tuples:
            if tail_detection not in mapped_tail and head_detection not in mapped_head:
                # if score > 20:
                # print("ReID Distance:", score, "<>", tail_detection.track_index)
                head_detection.track_index = tail_detection.track_index
                mapped_head.add(head_detection)
                mapped_tail.add(tail_detection)

        for head_detection in head.detections:
            if head_detection.track_index is None:
                head_detection.track_index = self.next_track_label()

    def process_bulk(self, bulk):
        if self.last_bulk is None:
            self.last_bulk = bulk
            for detection in bulk.head_frame.detections:
                detection.track_index = self.next_track_label()
            return

        last_head = self.last_bulk.head_frame
        if len(self.last_bulk.tail_frames) > 0:
            last_tail = self.last_bulk.tail_frames[-1]
        else:
            last_tail = self.last_bulk.head_frame

        new_head = bulk.head_frame

        self._match(last_head, last_tail, new_head)
        self.last_bulk = bulk
