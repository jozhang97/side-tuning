from evkit.models.taskonomy_network import TaskonomyDecoder
from tlkit.utils import SINGLE_IMAGE_TASKS, TASKS_TO_CHANNELS, FEED_FORWARD_TASKS
import torch
import torch.nn.functional as F

def softmax_cross_entropy(inputs, target, weight=None, cache={}, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    cache['predictions'] = inputs
    cache['labels'] = target
    if len(target.shape) == 2:   # unsqueeze one-hot representation
        target = torch.argmax(target, dim=1)
    loss = F.cross_entropy(inputs, target, weight)
    # when working with 2D data, cannot use spatial weight mask, it becomes categorical/class
    return {'total': loss, 'xentropy': loss}

def heteroscedastic_normal(mean_and_scales, target, weight=None, cache={}, eps=1e-2):
    mu, scales = mean_and_scales
    loss = (mu - target)**2 / (scales**2 + eps) + torch.log(scales**2 + eps)
#     return torch.sum(weight * loss) / torch.sum(weight) if weight is not None else loss.mean()
    loss = torch.mean(weight * loss) / weight.mean() if weight is not None else loss.mean()
    return {'total': loss, 'nll': loss}

def heteroscedastic_double_exponential(mean_and_scales, target, weight=None, cache={}, eps=5e-2):
    mu, scales = mean_and_scales
    loss = torch.abs(mu - target) / (scales + eps) + torch.log(2.0 * (scales + eps))
    loss = torch.mean(weight * loss) / weight.mean() if weight is not None else loss.mean()
    return {'total': loss, 'nll': loss}

def weighted_mse_loss(inputs, target, weight=None, cache={}):
    losses = {}
    cache['predictions'] = inputs
    cache['labels'] = target
    if weight is not None:
#         sq = (inputs - target) ** 2
#         weightsq = torch.sum(weight * sq)
        loss = torch.mean(weight * (inputs - target) ** 2)/torch.mean(weight)
    else:
        loss = F.mse_loss(inputs, target)
    return {'total': loss, 'mse': loss}

weighted_l2_loss = weighted_mse_loss

def weighted_l1_loss(inputs, target, weight=None, cache={}):
    target = target.float()
    if weight is not None:
        loss = torch.mean(weight * torch.abs(inputs - target))/torch.mean(weight)
    else:
        loss = F.l1_loss(inputs, target)
    return {'total': loss, 'l1': loss}

def perceptual_l1_loss(decoder_path, bake_decodings):
    task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
    decoder = TaskonomyDecoder(TASKS_TO_CHANNELS[task], feed_forward=task in FEED_FORWARD_TASKS)
    checkpoint = torch.load(decoder_path)
    decoder.load_state_dict(checkpoint['state_dict'])
    decoder.cuda()
    decoder.eval()
    print(f'Loaded decoder from {decoder_path} for perceptual loss')
    def runner(inputs, target, weight=None, cache={}):
        # the last arguments are so we can 'cache' and pass the decodings outside
        inputs_decoded = decoder(inputs)
        targets_decoded = target if bake_decodings else decoder(target)
        cache['predictions'] = inputs_decoded
        cache['labels'] = targets_decoded
        if weight is not None:
            loss = torch.mean(weight * torch.abs(inputs_decoded - targets_decoded))/torch.mean(weight)
        else:
            loss = F.l1_loss(inputs_decoded, targets_decoded)
        return {'total': loss, 'perceptual_l1': loss}
    return runner


def perceptual_l2_loss(decoder_path, bake_decodings):
    task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
    decoder = TaskonomyDecoder(TASKS_TO_CHANNELS[task], feed_forward=task in FEED_FORWARD_TASKS)
    checkpoint = torch.load(decoder_path)
    decoder.load_state_dict(checkpoint['state_dict'])
    decoder.cuda()
    decoder.eval()
    print(f'Loaded decoder from {decoder_path} for perceptual loss')
    def runner(inputs, target, weight=None, cache={}):
        # the last arguments are so we can 'cache' and pass the decodings outside
        inputs_decoded = decoder(inputs)
        targets_decoded = target if bake_decodings else decoder(target)
        cache['predictions'] = inputs_decoded
        cache['labels'] = targets_decoded
        if weight is not None:
            loss = torch.mean(weight * (inputs_decoded - targets_decoded) ** 2)/torch.mean(weight)
        else:
            loss = F.mse_loss(inputs_decoded, targets_decoded)
        return {'total': loss, 'perceptual_mse': loss}
    return runner


def dense_softmax_cross_entropy_loss(inputs, targets, cache={}):  # these should be logits  (batch_size, n_class)
    batch_size, _ = targets.shape
    losses = {}
    losses['final'] = -1. * torch.sum(torch.softmax(targets.float(), dim=1) * F.log_softmax(inputs.float(), dim=1)) / batch_size
    losses['standard'] = losses['final']
    return losses

def dense_cross_entropy_loss_(inputs, targets):  # these should be logits  (batch_size, n_class)
    batch_size, _ = targets.shape
    return -1. * torch.sum(targets * F.log_softmax(inputs, dim=1)) / batch_size

# def dense_softmax_cross_entropy(inputs, targets, weight=None, cache={}):
#     assert weight == None
#     cache['predictions'] = inputs
#     cache['labels'] = targets
#     # print(targets.shape)
#     batch_size, _ = targets.shape
#     loss =  -1. * torch.sum(torch.softmax(targets, dim=1) * F.log_softmax(inputs, dim=1)) / batch_size
#     loss =  F.mse_loss(inputs, targets.detach())
#     return {'total': loss, 'xentropy': loss}

def dense_softmax_cross_entropy(inputs, targets, weight=None, cache={}):
    assert weight is None
    cache['predictions'] = inputs
    cache['labels'] = targets

    batch_size, _ = targets.shape
    loss =  -1. * torch.sum(torch.softmax(targets.detach(), dim=1) * F.log_softmax(inputs, dim=1)) / batch_size
    # loss =  F.mse_loss(inputs, targets.detach())
    return {'total': loss, 'xentropy': loss}

def dense_cross_entropy(inputs, targets, weight=None, cache={}):
    assert weight == None
    cache['predictions'] = inputs
    cache['labels'] = targets

    batch_size, _ = targets.shape
    loss =  -1. * torch.sum(targets.detach() * F.log_softmax(inputs, dim=1)) / batch_size
    # loss =  F.mse_loss(inputs, targets.detach())
    return {'total': loss, 'xentropy': loss}

def perceptual_cross_entropy_loss(decoder_path, bake_decodings):
    task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
    decoder = TaskonomyDecoder(TASKS_TO_CHANNELS[task], feed_forward=task in FEED_FORWARD_TASKS)
    checkpoint = torch.load(decoder_path)
    decoder.load_state_dict(checkpoint['state_dict'])
    decoder.cuda()
    decoder.eval()
    print(f'Loaded decoder from {decoder_path} for perceptual loss')
    def runner(inputs, target, weight=None, cache={}):
        # the last arguments are so we can 'cache' and pass the decodings outside
        inputs_decoded = decoder(inputs)
        targets_decoded = target if bake_decodings else decoder(target)
        cache['predictions'] = inputs_decoded
        cache['labels'] = targets_decoded
        return dense_softmax_cross_entropy_loss_(inputs_decoded, targets_decoded)
    return runner

def identity_regularizer(loss_fn, model):
    def runner(inputs, target, weight=None, cache={}):
        losses = loss_fn(inputs, target, weight, cache)
        return losses
    return runner

def transfer_regularizer(loss_fn, model, reg_loss_fn='F.l1_loss', coef=1e-3):
    def runner(inputs, target, weight=None, cache={}):
        orig_losses = loss_fn(inputs, target, weight, cache)

        #if isinstance(model, PolicyWithBase):
        if type(model).__name__ == "PolicyWithBase":
            # Imitation Learning - retreive encodings via the cache
            assert 'base_encoding' in cache and 'transfered_encoding' in cache, f'cache is missing keys {cache.keys()}'
            regularization_loss = 0
            for base_encoding, transfered_encoding in zip(cache['base_encoding'], cache['transfered_encoding']):
                regularization_loss += eval(reg_loss_fn)(model.base.perception_unit.sidetuner.net.transfer_network(base_encoding), transfered_encoding)
        else:
            # Vision Transfers - retreive encodings directly from model attributes
            # (cannot do this for IL due to the FrameStacked being iterative)
            assert isinstance(model.side_output, torch.Tensor), 'Cannot regularize side network if it is not used'
            regularization_loss = eval(reg_loss_fn)(model.transfer_network(model.base_encoding), model.transfered_encoding)

        orig_losses.update({
            'total': orig_losses['total'] + coef * regularization_loss,
            'weight_tying': regularization_loss,
        })
        return orig_losses

    return runner

def perceptual_regularizer(loss_fn, model, coef=1e-3, decoder_path=None, use_transfer=True, reg_loss_fn='F.mse_loss'):
    # compares model.base_encoding E(x) and model.transfered_encoding T(E(x) + S(x))
    # use_transfer means we will compare exactly above
    # use_transfer=False means we will compare model.base_encoding E(x) and model.merged_encoding E(x) + S(x)
    # Recall, decoder requires unnormalized inputs!
    assert decoder_path is not None, 'Pass in a decoder to which to transform our parameters and regularize on'
    task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
    decoder = TaskonomyDecoder(TASKS_TO_CHANNELS[task], feed_forward=task in FEED_FORWARD_TASKS)
    checkpoint = torch.load(decoder_path)
    decoder.load_state_dict(checkpoint['state_dict'])
    decoder.cuda()
    decoder.eval()
    if task in FEED_FORWARD_TASKS:
        reg_loss_fn = "dense_softmax_cross_entropy_loss_"
    else:
        reg_loss_fn = "F.l1_loss"
    print(f'Loaded decoder from {decoder_path} for perceptual loss')

    def runner(inputs, target, weight=None, cache={}):
        orig_losses = loss_fn(inputs, target, weight, cache)

        if type(model).__name__ == "PolicyWithBase":
            # Imitation Learning - retreive encodings via the cache
            assert 'base_encoding' in cache, f'cache is missing base {cache.keys()}'
            if use_transfer:
                assert 'transfered_encoding' in cache, f'cache is missing tied {cache.keys()}'
                tied_encodings = cache['transfered_encoding']
            else:
                assert 'merged_encoding' in cache, f'cache is missing tied{cache.keys()}'
                tied_encodings = cache['merged_encoding']

            regularization_loss = 0
            for base_encoding, tied_encoding in zip(cache['base_encoding'], tied_encodings):
                regularization_loss += eval(reg_loss_fn)(decoder(base_encoding), decoder(tied_encoding))
        else:
            # Vision Transfers - retreive encodings directly from model attributes
            # (cannot do this for IL due to the FrameStacked being iterative)
            assert isinstance(model.side_output, torch.Tensor), 'Cannot regularize side network if it is not used'
            if use_transfer:
                tied_encoding = model.transfered_encoding
            else:
                tied_encoding = model.merged_encoding
            losses['weight_tying'] = eval(reg_loss_fn)(decoder(model.base_encoding), decoder(tied_encoding))
            regularization_loss = reg_loss_fn(decoder(model.base_encoding), decoder(tied_encoding))

        orig_losses.update({
            'total': orig_losses['total'] + coef * regularization_loss,
            'weight_tying': regularization_loss,
        })
        return orig_losses
    return runner

