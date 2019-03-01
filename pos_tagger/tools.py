import os
import time
import uuid
import logging


class Tools:
    def __init__(self):
        self.log_path = '/home/m.domrachev/repos/MorphemRuEval2019/pos_tagger/'

    def read_file(self, path2data_file):
        strings = []
        file_object = open(path2data_file, "r")
        for line in file_object.readlines():
            line = line.strip()
            strings.append(line)
        return strings

    def init_logging(self, file_name):
        fmt = logging.Formatter('%(asctime)-15s %(message)s')

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        log_dir_name = os.path.join(self.log_path, 'logs')
        log_file_name = time.strftime("%Y_%m_%d-%H_%M_%S-") + str(uuid.uuid4())[:8] + '_%s.txt' % (file_name, )
        logging.info('Logging to {}'.format(log_file_name))
        logfile = logging.FileHandler(os.path.join(log_dir_name, log_file_name), 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
        return log_dir_name
