import configargparse
import importlib
import codecs
from itertools import groupby

misc = importlib.import_module("OpenNMT-py.onmt.utils.misc")
translator = importlib.import_module("OpenNMT-py.onmt.translate.translator")
opts = importlib.import_module("OpenNMT-py.onmt.opts")


class Lemmatizer:
    """
    Лемматизатор работает следующим образом:

    на вход файл формата:

    _	əmərə	_	VERB	_	_	_	_	_	_
    _	dəː	_	PART	_	_	_	_	_	_
    _	uďaliːn	_	NOUN	_	_	_	_	_	_

    _	buː	_	PRON	_	_	_	_	_	_
    _	pəktirulləw	_	VERB	_	_	_	_	_	_


    на выходе файл формата:

    _	əmərə	əmə	VERB	_	_	_	_	_	_
    _	dəː	dəː	PART	_	_	_	_	_	_
    _	uďaliːn	uďa	NOUN	_	_	_	_	_	_

    _	buː	buː	PRON	_	_	_	_	_	_
    _	pəktirulləw	pəktiru	VERB	_	_	_	_	_	_

    """

    def __init__(self, opt):
        self.opt = opt
        self.max_sent_len = 0
        self.translator = translator.build_translator(self.opt, report_score=True)
        self.input_data = self.form_input()
        print('Translator ready.')

    def count_sent(self, path):
        with codecs.open(path, "r", encoding="utf-8") as f:
            sentences = [[t.split('\t') for t in list(y)] for x, y in groupby(f.readlines(), lambda z: z == '\n') if
                         not x]
            return len(sentences), max([len(s) for s in sentences])

    def split_lemmatize_corpus(self, path):
        with codecs.open(path, "r", encoding="utf-8") as f:
            sentences = [[t.split('\t') for t in list(y)] for x, y in groupby(f.readlines(), lambda z: z == '\n') if
                         not x]
            sentences = [[' '.join(list(t[1])) + ' ' + t[3] for t in s] for s in sentences if s != ['\n']]
            while True:
                for s in sentences:
                    yield s

    def form_input(self):
        src_shards = self.split_lemmatize_corpus(self.opt.src)
        c_s, max_len_s = self.count_sent(self.opt.src)
        self.max_sent_len = max_len_s
        tgt_shards = [None] * c_s
        return zip(src_shards, tgt_shards)

    def lemmatize(self):
        for i, (src_shard, tgt_shard) in enumerate(self.input_data):
            self.translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=self.opt.src_dir,
                batch_size=self.max_sent_len,
                attn_debug=self.opt.attn_debug,
                use_sent_borders=True
            )


if __name__ == '__main__':

    parser = configargparse.ArgumentParser(
        description='lemmatizer.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args()

    lemmatizator = Lemmatizer(opt)
    lemmatizator.lemmatize()
