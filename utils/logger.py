import logging
import datetime


def config_logger(log_level, main_filename):
    """Configures the logger.

    Only call this once per run because can only configure once. Call this
    before making any LogRecords. Saves log file as same base name as the main
    script.

    Note: There are 5 levels of severity (in increasing order),
        DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    logleveldict = {'DEBUG': logging.DEBUG,
                    'INFO': logging.INFO,
                    'WARNING': logging.WARNING,
                    'ERROR': logging.ERROR,
                    'CRITICAL': logging.CRITICAL}

    log_file = main_filename.split('.')[0] + '.log'  # Name of log file

    # Set options for logging
    logging.basicConfig(level=logleveldict[log_level],
                        filename=log_file,
                        filemode='a',
                        format='%(message)s')


def log_start(config):
    """Logs current configs."""
    logging.info('')
    logging.info('')
    logging.info('')
    logging.info('---------- Start ----------')
    logging.info(f'Date Time: {datetime.datetime.now()}')
    logging.info(f'Using config file: {config.json_file}')
    logging.info(f'Experiment name: {config.experiment_name}')
    logging.info(f'Dataset: {config.data_loader.name}')
    logging.info(f'Reading data from: {config.data_loader.data_dir}')
    logging.info(f'Batch size: {config.data_loader.batch_size}')
    logging.info(f'Shuffle training data: {config.data_loader.shuffle}')
    logging.info(f'Num workers loading data: {config.data_loader.num_workers}')
    logging.info(f'Optimizer: {config.optimizer.name}')
    logging.info(f'Learning rate: {config.optimizer.learning_rate}')
    logging.info(f'Loss function: {config.loss_function}')
    logging.info(f'Num epochs to train to: {config.trainer.epochs}')


def log_end():
    logging.info('')
    logging.info('---------- End ----------')
    logging.info(f'Date Time: {datetime.datetime.now()}')
