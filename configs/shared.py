@ex.named_config
def radam():
    cfg = {}
    cfg['learner'] = {
        'optimizer_class': 'RAdam'
    }

@ex.named_config
def reckless():
    cfg = {}
    cfg['training'] = {
        'resume_training': False,
    }
    cfg['saving'] = {
        'obliterate_logs': True,
    }
