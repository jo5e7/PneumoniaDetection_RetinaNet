from pneumonia.visualize import predict

test_path = "/home/jdmaestre/PycharmProjects/Pneumonia_dataset/jpg_minitest"
model_path =  "/home/jdmaestre/PycharmProjects/pytorch-retinanet/model_final.pt"


predict(test_path, model_path)
