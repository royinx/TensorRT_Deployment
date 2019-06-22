from .._deprecated_helpers import *
import tensorrt as trt

warn_deprecated("The parsers submodule will been removed in a future version of the TensorRT Python API")

# Expose "sub-modules"
# TODO: PluginFactory/PluginFactory Ext Class Definitions
class caffeparser(object):
    CaffeParser = trt.CaffeParser
    CaffeParser.destroy = deprecated(lambda this: this.__del__(), use="del parser")
    deprecated_setter(CaffeParser, "plugin_factory")
    deprecated_setter(CaffeParser, "plugin_factory_ext")
    deprecated_setter(CaffeParser, "protobuf_buffer_size")
    CaffeParser.shutdown_protobuf_library = deprecated(trt.shutdown_protobuf_library, use="trt.shutdown_protobuf_library")
    shutdown_protobuf_library = deprecated(trt.shutdown_protobuf_library, use="trt.shutdown_protobuf_library")
    BlobNameToTensor = trt.IBlobNameToTensor
    create_caffe_parser = trt.CaffeParser

    class BinaryProtoBlob(object):
        def __init__(self, nparray):
            self.arr = nparray

        def get_data(self, *args):
            warn_deprecated("parse_binary_proto returns a numpy array. Therefore, the get_data function is no longer required.")
            return self.arr

        def get_data_type(self, *args):
            warn_deprecated("parse_binary_proto returns a numpy array. Therefore, the get_data function is no longer required.")
            return self.arr.dtype

        def get_dimensions(self, *args):
            warn_deprecated("parse_binary_proto returns a numpy array. Therefore, the get_data function is no longer required.")
            return self.arr.shape

        def destroy(self):
            warn_deprecated("parse_binary_proto returns a numpy array. The BinaryProtoBlob is a thin compatibility wrapper around the numpy array and will be deprecated in future releases. Therefore, this destroy function has no effect.")

    CaffeParser._parse_binary_proto_impl = trt.CaffeParser.parse_binary_proto
    CaffeParser.parse_binary_proto = lambda this, proto: caffeparser.BinaryProtoBlob(this._parse_binary_proto_impl(proto))
    PluginFactory = trt.ICaffePluginFactory

class uffparser(object):
    UffParser = trt.UffParser
    UffParser.destroy = deprecated(lambda this: this.__del__(), use="del parser")
    deprecated_getter(UffParser, "uff_required_version_major")
    deprecated_getter(UffParser, "uff_required_version_minor")
    deprecated_getter(UffParser, "uff_required_version_patch")
    deprecated_setter(UffParser, "plugin_factory")
    deprecated_setter(UffParser, "plugin_factory_ext")
    # Registering inputs and outputs and avoid infinte recursion
    UffParser._register_input_internal = trt.UffParser.register_input
    UffParser.register_input = lambda this, name, shape, order=0: this._register_input_internal(name, shape, trt.UffInputOrder(order) if type(order) == int else order)
    # Set parsing stuff properly.
    UffParser.parse_from_file = UffParser.parse
    UffParser._parse_buffer_impl = trt.UffParser.parse_buffer
    UffParser.parse_buffer = deprecated(lambda this, buf, size, network, weights_type=trt.float32 : this._parse_buffer_impl(buf, network, weights_type), use="parse_buffer without providing size (size is automatically deduced)")

    UffParser.shutdown_protobuf_library = deprecated(trt.shutdown_protobuf_library, use="trt.shutdown_protobuf_library")
    shutdown_protobuf_library = deprecated(trt.shutdown_protobuf_library, use="trt.shutdown_protobuf_library")
    create_uff_parser = trt.UffParser
    PluginFactory = trt.IUffPluginFactory

class onnxparser(object):
    ONNXParser = trt.OnnxParser
    ONNXParser.destroy = deprecated(lambda this: this.__del__(), use="del onnx_parser")
    create_onnxparser = lambda network, logger: trt.OnnxParser(network, logger)
