from __future__ import print_function, division


def print_and_log(string, logger):
    print(string)
    if logger:
        logger.info(string)
