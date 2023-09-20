config_294 = {
    'name': 'mf294',
    'token': 6,  # tokens
    'embed': 128,  # embed_dim
    'stem': 16,
    # stage1
    'bneck': {'e': 32, 'o': 16, 's': 1},  # exp out stride
    'body': [
        # stage2
        {'inp': 16, 'exp': 96, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 24, 'exp': 96, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
        # stage3
        {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        # stage4
        {'inp': 48, 'exp': 288, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 96, 'exp': 384, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 96, 'exp': 576, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
    ],
}

config = {
    'mf294': config_294,
}
