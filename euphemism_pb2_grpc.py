# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import euphemism_pb2 as euphemism__pb2


class EuphemismServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DetectEuphemism = channel.unary_unary(
                '/EuphemismService/DetectEuphemism',
                request_serializer=euphemism__pb2.DetectEuphemismRequest.SerializeToString,
                response_deserializer=euphemism__pb2.DetectEuphemismResponse.FromString,
                )


class EuphemismServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def DetectEuphemism(self, request, context):
        """检验句子是否存在委婉语
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EuphemismServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DetectEuphemism': grpc.unary_unary_rpc_method_handler(
                    servicer.DetectEuphemism,
                    request_deserializer=euphemism__pb2.DetectEuphemismRequest.FromString,
                    response_serializer=euphemism__pb2.DetectEuphemismResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'EuphemismService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class EuphemismService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def DetectEuphemism(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/EuphemismService/DetectEuphemism',
            euphemism__pb2.DetectEuphemismRequest.SerializeToString,
            euphemism__pb2.DetectEuphemismResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)