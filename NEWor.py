from oneNeuron.perceptron import Perceptron
# from utils.model import Perceptron
from utils.common_utils import prepare_data, save_plot, save_model
import pandas as pd
import logging
import os

logging_str = "[%(asctime)s - %(levelname)s - %(module)s] %(message)s"
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logs_dir,"running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")


def main(data, modelname, plotfilename, eta, epochs):
    df = pd.DataFrame(data)
    logging.info(f"input dataframe : \n {df}")

    X,y = prepare_data(df)
    model = Perceptron(eta, epochs)
    model.fit(X,y)
    _ = model.total_loss()
    save_model(model, modelname)
    save_plot(df , plotfilename, model)

if __name__ == '__main__':
    OR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,1]   
    }
    ETA = 0.03
    EPOCHS = 10
    try:
        logging.info(">>>>> started the model training >>>>>")
        main(data=OR, modelname="or.model", plotfilename="NEWor.png", eta = ETA, epochs=EPOCHS) 
        logging.info("<<<<< model training done successfully <<<<< \n")
    except Exception as e:
        logging.exception(e)
        raise e