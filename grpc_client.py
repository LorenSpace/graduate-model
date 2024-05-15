import grpc

import euphemism_pb2_grpc, euphemism_pb2


def run_client():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = euphemism_pb2_grpc.EuphemismServiceStub(channel)
        response = stub.DetectEuphemism(euphemism_pb2.DetectEuphemismRequest(text="I am <happy> to see you."))
        print("Euphemism detected:", response.result)


if __name__ == '__main__':
    run_client()
