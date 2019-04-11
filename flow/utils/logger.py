from pathlib import Path
import logging
import datetime
import __main__


def configure_logger(config, log_to_screen=False):
    """Configures the logger.

    Only call this once per run because can only configure once. Call this
    before making any LogRecords. Saves log file with same base name as the
    main script.

    Note: There are 5 levels of severity (in increasing order),
        DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    logleveldict = {'DEBUG': logging.DEBUG,
                    'INFO': logging.INFO,
                    'WARNING': logging.WARNING,
                    'ERROR': logging.ERROR,
                    'CRITICAL': logging.CRITICAL}

    if log_to_screen:
        # Prints to screen
        log_file = None
    else:
        # Log file same base name as the main file
        log_file = config.results_dir / config.main_file.name
        log_file = log_file.with_suffix('.log')

    # Set options for logging
    logging.basicConfig(level=logleveldict[config.log_level],
                        filename=log_file,
                        filemode='a',
                        format='%(message)s')


def log_start(config):
    """Logs current configs."""
    logging.info('')
    logging.info('')
    logging.info('')
    logging.info('---------- START ----------')
    logging.info(f'Date Time: {datetime.datetime.now()}')
    logging.info('')
    logging.info(f'Using config file: {config.json_file}')
    logging.info(f'Experiment directory: {config.experiment_dir}')
    logging.info(f'Dataset: {config.data_loader.name}')
    logging.info(f'Reading data from: {config.data_loader.data_dir}')
    logging.info(f'Batch size: {config.data_loader.batch_size}')
    logging.info(f'Shuffle training data: {config.data_loader.shuffle}')
    logging.info(f'Num workers loading data: {config.data_loader.num_workers}')
    logging.info(f'Optimizer: {config.optimizer.name}')
    logging.info(f'Learning rate: {config.optimizer.learning_rate}')
    logging.info(f'Loss function: {config.loss_function}')
    logging.info(f'Num epochs to train to: {config.trainer.max_epoch}')


def log_end():
    logging.info('')
    logging.info('---------- END ----------')
    logging.info(f'Date Time: {datetime.datetime.now()}')
