import spacy

# nlp = spacy.blank("en")
#
#
# def word_tokenize(sent):
#     doc = nlp(sent)
#     return [token.text for token in doc]
#
# def convert_idx(text, tokens):
#     current = 0
#     spans = []
#     for token in tokens:
#         current = text.find(token, current) # 字符串查找S.find(sub, start=None, end=None),返回被找到的第一个sub下标
#         if current < 0:
#             print("Token {} cannot be found".format(token))
#             raise Exception()
#         spans.append((current, current + len(token)))
#         current += len(token)
#     return spans
#
a = ' i love zzz very much'
# context_tokens = word_tokenize(a)
# # print(a.replace('q','22').replace("w","11"))
# print(convert_idx(a, context_tokens))
array = a.split()
word = "".join(array[0:-4])
print(word)
