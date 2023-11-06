#!/usr/bin/python
import logging.config
import yaml
import logging
import codecs


class Logger:

    with codecs.open('config.yaml', 'r', encoding="utf-8") as f:
        log_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(log_cfg)
    logger_dev = logging.getLogger('dev')
    logger_dev.propagate =  False
    logger_test = logging.getLogger("test")

    def set_level(logger, level):
        logger.setLevel(level)


if __name__ == "__main__":
    def test():
        Logger.set_level(Logger.logger_dev,logging.INFO)
        Logger.logger_dev.info(" this is the correct information")
        Logger.logger_dev.shutdown()
    test()