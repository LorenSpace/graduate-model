# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: euphemism.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0f\x65uphemism.proto\"&\n\x16\x44\x65tectEuphemismRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\")\n\x17\x44\x65tectEuphemismResponse\x12\x0e\n\x06result\x18\x01 \x01(\x08\x32Z\n\x10\x45uphemismService\x12\x46\n\x0f\x44\x65tectEuphemism\x12\x17.DetectEuphemismRequest\x1a\x18.DetectEuphemismResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'euphemism_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_DETECTEUPHEMISMREQUEST']._serialized_start=19
  _globals['_DETECTEUPHEMISMREQUEST']._serialized_end=57
  _globals['_DETECTEUPHEMISMRESPONSE']._serialized_start=59
  _globals['_DETECTEUPHEMISMRESPONSE']._serialized_end=100
  _globals['_EUPHEMISMSERVICE']._serialized_start=102
  _globals['_EUPHEMISMSERVICE']._serialized_end=192
# @@protoc_insertion_point(module_scope)
