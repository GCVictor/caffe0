add_library(protolib caffe.proto)
protobuf_generate(TARGET protolib LANGUAGE cpp PROTOC_OUT_DIR ${PROTO_GEN_DIR})
