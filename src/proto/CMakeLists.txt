set(PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

add_library(protolib caffe.proto)
protobuf_generate(TARGET protolib LANGUAGE cpp PROTOC_OUT_DIR ${PROTO_GEN_DIR})
