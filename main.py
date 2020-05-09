import argparse

from train import prepare_data, train_and_save_model
from real_time_inference import predict_real_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Neural Network Models")
    parser.add_argument("--command", required=True, type=str)
    args = parser.parse_args()

    if args.command == "train_conv1":
        x_tr, x_val, y_tr, y_val = prepare_data(True)
        train_and_save_model(1, x_tr, x_val, y_tr, y_val)

    if args.command == "train_conv2":
        x_tr, x_val, y_tr, y_val = prepare_data(True)
        train_and_save_model(2, x_tr, x_val, y_tr, y_val)

    if args.command == "train_GRU":
        x_tr, x_val, y_tr, y_val = prepare_data(False)
        train_and_save_model(3, x_tr, x_val, y_tr, y_val)

    if args.command == "predict":
        model_num = int(input("Enter [1] for 1-Conv, [2] for 2-Conv, [3] for GRU: "))
        predict_real_time(model_num)



