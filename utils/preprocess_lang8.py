import argparse
import json
import re
import unicodedata


invalid_bytes_re = re.compile(r'[\x00-\x1F]+')
sline_re = re.compile(r'\[sline\].*?\[/sline\]')
color_tags = ['[f-blue]','[/f-blue]',
              '[f-red]','[/f-red]',
              '[f-bold]','[/f-bold]']
ja_re = re.compile(r'([ぁ-んァ-ン])')
html_re = re.compile(r'<(\/?[a-z]+)>')
subsent_delim_re = re.compile(r'\.|。|\?|!|,|、|\(|\)')


def clean_sent(sent):
    sent = unicodedata.normalize('NFKC', sent.strip())
    for tag in color_tags:
        sent = sent.replace(tag, '')
    sent = sline_re.sub('', sent).replace('[/sline]', '')
    return sent


def check_sents(src_sent, tgt_sent):
    if src_sent == tgt_sent:
        return False
    if tgt_sent.endswith('OK') or tgt_sent.endswith('GOOD'):
        return False
    if not ja_re.search(tgt_sent) or html_re.search(tgt_sent):
        return False
    src_subsents = [x for x in subsent_delim_re.split(src_sent) if x]
    tgt_subsents = [x for x in subsent_delim_re.split(tgt_sent) if x]
    if len(src_subsents) != len(tgt_subsents):
        return False
    return True


def parse_lang8_line(line):
    row = json.loads(invalid_bytes_re.sub('', line))
    if row[2] != 'Japanese':
        return []
    pairs = set()
    for src_sent, tgt_sents in zip(row[4], row[5]):
        if not ja_re.search(src_sent) or html_re.search(src_sent):
            continue
        src_sent = clean_sent(src_sent)
        for tgt_sent in tgt_sents:
            if not tgt_sent:
                continue
            tgt_sent = clean_sent(tgt_sent)
            if not check_sents(src_sent, tgt_sent):
                continue
            pairs.add((src_sent, tgt_sent))
    return list(pairs)


def main(args):
    with open(args.corpus_path, encoding='utf-8') as f:
        lines = f.readlines()
    n = 0
    with open(args.src_lines_path, 'w', encoding='utf-8') as sf:
        with open(args.tgt_lines_path, 'w', encoding='utf-8') as tf:
            for line in lines:
                for src_sent, tgt_sent in parse_lang8_line(line):
                    sf.write(src_sent + '\n')
                    tf.write(tgt_sent + '\n')
                    n += 1
    print(f'Wrote {n} lines to parallel corpus output')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus_path', required=True,
                        help='Path to Lang8 corpus file')
    parser.add_argument('-s', '--src_lines_path', required=True,
                        help='Path to source lines in parallel corpus output')
    parser.add_argument('-t', '--tgt_lines_path', required=True,
                        help='Path to target lines in parallel corpus output')
    args = parser.parse_args()
    main(args)
