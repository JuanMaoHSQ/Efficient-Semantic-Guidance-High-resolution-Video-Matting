from ptflops import get_model_complexity_info
from model.model import MattingNetwork

model = MattingNetwork()
macs, params = get_model_complexity_info(model,(3,512,288),as_strings=True,print_per_layer_stat=True)
print("%s %s" % (macs,params))

