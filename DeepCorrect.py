import config
from construct_model_input import TrainDataset
from train_model import train, build_model
import logging.config
logger = logging.getLogger(__name__)


def main():
    train_dataset = TrainDataset(config.input_feature_file_train, config.input_spectrum_file_train)
    val_dataset = TrainDataset(config.input_feature_file_val, config.input_spectrum_file_train)
    model = train(train_dataset=val_dataset, val_dataset=val_dataset, model=build_model())
#    test(dataset, model)
#    test(dataset, build_model('model/0:19'))


if __name__ == '__main__':
    log_file_name = 'DeepNovo.log'
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }

    logging.config.dictConfig(d)
    main()