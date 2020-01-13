
from .actor_critic_module import NaivelyRecurrentACModule

   
class ForwardInverseACModule(NaivelyRecurrentACModule):
    ''' 
        This Module adds a forward-inverse model on top of the perception unit. 
    '''
    def __init__(self, perception_unit, forward_model, inverse_model, use_recurrency=False, internal_state_size=512):
        super().__init__(perception_unit, use_recurrency, internal_state_size)

        self.forward_model = forward_model
        self.inverse_model = inverse_model