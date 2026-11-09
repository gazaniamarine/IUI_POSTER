import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst

Gst.init(None)


class CameraStream:
    def __init__(self, port=5000):
        self.pipeline = Gst.parse_launch(
            f'udpsrc port={port} buffer-size=262144 '
            'caps="application/x-rtp,media=video,encoding-name=JPEG,'
            'payload=26,clock-rate=90000" ! '
            'rtpjitterbuffer latency=0 drop-on-latency=true ! '
            'rtpjpegdepay ! '
            'queue max-size-buffers=1 leaky=downstream ! '
            'jpegdec ! '
            'queue max-size-buffers=1 leaky=downstream ! '
            'videoconvert ! '
            'video/x-raw,format=BGR ! '
            'appsink name=sink sync=false max-buffers=1 drop=true'
        )

        self.sink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)

    def read(self):
        sample = self.sink.emit("try-pull-sample", 0)

        if sample is None:
            return None

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)

        if not success:
            return None

        caps = sample.get_caps()
        structure = caps.get_structure(0)

        width = structure.get_value("width")
        height = structure.get_value("height")

        frame = np.frombuffer(
            map_info.data,
            dtype=np.uint8
        ).reshape((height, width, 3))

        buffer.unmap(map_info)

        return frame.copy()

    def release(self):
        self.pipeline.set_state(Gst.State.NULL)