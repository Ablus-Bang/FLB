from proto_py import communicate_pb2_grpc, communicate_pb2
import os
import io
import torch
import json
from datetime import datetime


def deserialize_model_state_dict(serialized_state_dict):
    state_dict = {}
    for key, byte_tensor in serialized_state_dict.items():
        # Use a BytesIO object to load the serialized tensor back into a tensor
        buffer = io.BytesIO(byte_tensor)
        tensor = torch.load(buffer)
        state_dict[key] = tensor
    return state_dict


class WeightsTransferServicer(communicate_pb2_grpc.WeightsTransferServicer):
    def __init__(self, base_server):
        self.base_server = base_server

    def SendWeights(self, request, context):
        if request.HasField('send_parameters'):
            client_id = request.send_parameters.client_id
            train_dataset_length = request.send_parameters.train_dataset_length
            new_model_weight = request.send_parameters.new_model_weight
            # Process the received weights here
            print(f"Received weights from client {client_id} with dataset length {train_dataset_length}")
            # save client weights
            current_date = datetime.today().strftime("%Y%m%d_%H%M%S")
            client_save_path = os.path.join(
                self.base_server.save_path,
                "local_output_{}".format(str(client_id)),
                current_date.split('_')[0],
                current_date.split('_')[1],
            )
            os.makedirs(client_save_path, exist_ok=True)
            torch.save(
                deserialize_model_state_dict(new_model_weight),
                client_save_path + "/pytorch_model.bin",
            )
            with open(client_save_path + '/train_dataset_length.json', 'w') as f:
                json.dump({"train_dataset_length": train_dataset_length}, f)
            # Return a successful response
            return communicate_pb2.TransferStatus(code=True, message="Weights received successfully")
        else:
            return communicate_pb2.TransferStatus(code=False, message="Invalid message type")
