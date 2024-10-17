import joblib

def svm_inference(iteration):
    # Load the model from the file
    loaded_svm = joblib.load('/home/jiaxi/cs525/Experiment/prediction_model/svm/svm_model.pkl')

    sample = [[5.0, 3.6, 1.4, 0.2]]
    for i in range(iteration):
        loaded_svm.predict(sample)