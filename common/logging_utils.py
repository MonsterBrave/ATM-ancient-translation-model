import logging

def get_logger(name, level):
    logger = logging.getLogger(name)
    
    if level == 'info':
        logger.setLevel(logging.INFO)
    if level == 'debug':
        logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s-%(asctime)s-%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger