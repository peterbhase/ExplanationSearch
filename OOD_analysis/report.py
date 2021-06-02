import sys
import pandas as pd

class Report():
    """
    Report stores evaluation results during the training process as text files.
    Saves results for grid search in args.result_sheet_name
    """

    def __init__(self, args, file_path, experiment_name, score_names):
        self.fn = file_path
        self.args = args
        self.text = ''
        self.max_len = 10

        # write input arguments at the top
        self.text += 'Input: python %s %s \n\n' % \
                         (sys.argv[0], 
                          ' '.join([arg for arg in sys.argv[1:]]))

        # make header
        header = '%6s |' % 'epoch'
        for n, score_name in enumerate(score_names):
            len_name = len(score_name)
            if len_name > self.max_len:
                score_name = score_name[:self.max_len]
            header += (' %10s' % score_name)
            if n < len(score_names) - 1: header += '|'
        self.header = header

        # write header
        self.blank_line = '-'*len(header)
        self.text += self.blank_line + \
                    f"\nTraining report for model: {experiment_name}" + \
                    '\n' + self.blank_line + \
                    '\n'
        self.text += header


    def write_epoch_scores(self, epoch, scores):
        # write scores
        self.text += '\n%6s |' % str(epoch)
        for n, score in enumerate(scores.values()):
            self.text += ' %10s' % ('%1.2f' % score)
            if n < len(scores) - 1: self.text += '|'
        self.__save()

    def write_final_score(self, args, final_score_str, time_msg):
        self.text += '\n' + self.blank_line
        self.text += '\n%s' % final_score_str
        self.text += '\n' + self.blank_line + '\n'
        
        self.text += '\n'
        if time_msg is not None:
            self.text += f'\n\n{time_msg}\n\n'
        self._write_all_arguments(args)

        self.__save()

    def write_msg(self, msg):
        self.text += self.blank_line
        self.text += msg
        self.__save()

    def _write_all_arguments(self, args):
        self.text += "\nAll arguments:\n"
        self.text += '\n'.join(['\t' + hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])        
        self.__save()

    def print_epoch_scores(self, epoch, scores):
        epoch_text = ' %6s |' % 'epoch'
        for n, score_name in enumerate(scores.keys()):
            len_name = len(score_name)
            if len_name > self.max_len:
                score_name = score_name[:self.max_len]
            epoch_text += ' %10s' % score_name
            if n < len(scores) - 1: epoch_text += '|'
        epoch_text += '\n %6s |' % ('%d'% epoch)
        for n, score in enumerate(scores.values()):
            epoch_text += ' %10s' % ('%1.2f' % score)
            if n < len(scores) - 1: epoch_text += '|'
        print(epoch_text)

    def full_print(self):
        print('\n' + self.text + '\n')

    def __save(self):
        if self.fn is not None:
            with open(self.fn, mode='w') as text_file:
                text_file.write(self.text)

