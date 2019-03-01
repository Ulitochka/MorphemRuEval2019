from models.modules.layers.encoders import *
from models.modules.layers.decoders import *


released_models = {
    "BertBiLSTMNCRF": {
        "encoder": BertBiLSTMEncoder,
        "decoder": NCRFDecoder
    }
}
