import regex
import tiktoken

enc = tiktoken.get_encoding("gpt2")

# add '(?:[sdmt]|ll|ve|re)\p{L}+ to fix the issue with "'regular" -> "'re" + "gular"; "regular" -> regular
pat_str =  r"""'(?:[sdmt]|ll|ve|re)\p{L}+| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pat = regex.compile(pat_str)

def get_words_for_text(text):
  words = pat.findall(text)
  return words

def get_char_to_tokens_for_word(word):
    word_bytes = word.encode('utf-8')
    char2tokens = []
    for i in range(len(word_bytes) + 1):
        if i > 0:
          left = enc._encode_single_piece(word_bytes[:i])
        else:
          left = []
        right = []
        for j in range(i + 1, len(word_bytes) + 1):
            if j > i + 130:
                break
            if word_bytes[i:j] in enc._mergeable_ranks:
                right.append(enc._mergeable_ranks[word_bytes[i:j]])
        char2tokens.append([left[-1] if left else -1, right])
    return char2tokens

def unite_lists(lists):
    if not lists:
      return []
    result = lists[0]
    for l in lists[1:]:
        result[-1][1] = l[0][1]
        result.extend(l[1:])
    return result

def get_char_to_tokens_for_text(text):
  words = get_words_for_text(text)
  char2tokens = [get_char_to_tokens_for_word(x) for x in words]
  char2tokens = unite_lists(char2tokens)
  return char2tokens

def get_char2trace_step(char2tokens):
  char2trace_step = [-1] # there is no trace for <|endoftext|>, mask will be 0 for this token
  for i in range(1, len(char2tokens)):
    c = char2tokens[i][0]
    token_len = len(enc.decode_bytes([c]))
    prev_i = i - token_len
    assert c in char2tokens[prev_i][1]
    char2trace_step.append(prev_i)
  return char2trace_step

def get_char2trace(char2trace_step, init_trace=[0]):
  start_pos = init_trace[-1]
  char2trace = [init_trace] # for <|endoftext|> trace contains only this token
  for i, prev_i in enumerate(char2trace_step[1:], start=1):
    trace = list(char2trace[prev_i]) + [start_pos + i] # update trace with current token
    char2trace.append(trace)
  return char2trace