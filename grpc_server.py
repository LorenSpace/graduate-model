import argparse
import logging
import grpc
from concurrent import futures
import euphemism_pb2
import euphemism_pb2_grpc
import input_test

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EuphemismServiceServicer(euphemism_pb2_grpc.EuphemismServiceServicer):
    """服务实现检测委婉语"""

    def __init__(self, args):
        self.args = args

    def DetectEuphemism(self, request: euphemism_pb2.DetectEuphemismRequest,
                        context) -> euphemism_pb2.DetectEuphemismResponse:
        try:
            # 这里添加您的逻辑来检测委婉语
            result = input_test.detect_euphemism(request.text, self.args)
            return euphemism_pb2.DetectEuphemismResponse(result=result)
        except Exception as e:
            logger.error("Error detecting euphemism: %s", e)
            context.set_details("Internal server error")
            context.set_code(grpc.StatusCode.INTERNAL)
            return euphemism_pb2.DetectEuphemismResponse()


def serve(args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    euphemism_pb2_grpc.add_EuphemismServiceServicer_to_server(EuphemismServiceServicer(args), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Server running on port 50051...")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='roberta-large')
    parser.add_argument("--model_type", type=str, default='pet')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    args.pet_dim = 1024 if "large" in args.model else 768

    serve(args)
