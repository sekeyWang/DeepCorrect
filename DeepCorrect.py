import config
from construct_model_input import TrainDataset
from data_reader import DenovoDataset
from train_model import train, build_model, test, analyze_threshold, analyze_result, analyze_distribution
import logging.config
#from output_result import output_score
logger = logging.getLogger(__name__)


def main():
    train_dataset = TrainDataset(DenovoDataset(config.input_feature_file_train, config.input_spectrum_file_train)) + \
                    TrainDataset(DenovoDataset(config.input_feature_file_train, config.input_spectrum_file_train, real_peptide= True))
    val_dataset = TrainDataset(DenovoDataset(config.input_feature_file_val, config.input_spectrum_file_train))
    test_dataset = TrainDataset(DenovoDataset(config.input_feature_file_test, config.input_spectrum_file_train))

#    model = train(train_dataset=train_dataset, val_dataset=val_dataset, model=build_model())

    output_list, target_list = test(test_dataset, build_model("model/model3-6"))
    analyze_result(output_list, target_list)
#    output_score(config.input_feature_file_test, config.output_file_test, output_list, target_list)
#    analyze_distribution(output_list, target_list)
#    analyze_threshold(output_list, target_list)
#    analyze_result(output_list, target_list, 60)


if __name__ == '__main__':
    log_file_name = 'DeepCorrect.log'
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