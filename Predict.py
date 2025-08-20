from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    #model_path = os.path.join("checkpoints", "589311", "589311-52500") # this is yen with 2-4 chennel all > -35dB no gain adjusted with wave mse loss
    #model_path = os.path.join("checkpoints", "200430", "200430-40700") # this is yen with 2-6 chennel all > -35dB no gain adjusted with wave mse loss
    #model_path = os.path.join("checkpoints", "797025", "797025-29100") # this is yen with 2-6 chennel all > -35dB no gain adjusted with wave mse loss 0.5 & perceptual loss
    #model_path = os.path.join("checkpoints", "849853", "849853-62800") # this is yen with 2-6 chennel all > -35dB no gain adjusted with wave mse loss 0.9 & perceptual loss
    model_path = os.path.join("checkpoints", "455880", "455880-48800") # this is yen with 2-6 chennel all > -35dB no gain adjusted with wave mse loss 0.8 & perceptual loss






    input_path = os.path.join("mixture.wav")
    output_path = "testoutput" # Where to save results. Default: Same location as input.

@ex.automain
def main(cfg, model_path, input_path, output_path):
    model_config = cfg["model_config"]
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path)