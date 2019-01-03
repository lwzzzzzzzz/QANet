import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--glove_char_size', type=int, default=94, help="Corpus size for Glove")
parser.add_argument('--glove_word_size', type=int, default=int(2.2e6), help="Corpus size for Glove")
parser.add_argument('--glove_dim', type=int, default=300, help="Embedding dimension for Glove")

args = parser.parse_args()

print(args.glove_char_size)
