from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata

ObjectDetectorWriter = object_detector.MetadataWriter
_MODEL_PATH = "/home/maxim/PycharmProjects/tflite/raspi_v1/models/duplo_efficientdet_lite0_edgetpu.tflite"
_LABEL_FILE = "/home/maxim/PycharmProjects/tflite/raspi_v1/models/label.txt"
_SAVE_TO_PATH = "/home/maxim/PycharmProjects/tflite/raspi_v1/models/duplo_efficientdet_lite0_edgetpu_meta.tflite"

writer = ObjectDetectorWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [127.5], [127.5], [_LABEL_FILE])
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)

# Verify the populated metadata and associated files.
displayer = metadata.MetadataDisplayer.with_model_file(_SAVE_TO_PATH)
print("Metadata populated:")
print(displayer.get_metadata_json())
print("Associated file(s) populated:")
print(displayer.get_packed_associated_file_list())