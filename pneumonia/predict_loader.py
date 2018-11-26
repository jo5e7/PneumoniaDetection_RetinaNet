from pneumonia.visualize import predict

test_path = "/home/jdmaestre/PycharmProjects/Pneumonia_dataset/jpg_test"
model_path =  "/home/jdmaestre/PycharmProjects/final_models/20ep_50res_5bs_syn/model_final.pt"


predict(test_path, model_path)
