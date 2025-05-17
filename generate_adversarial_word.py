from Pinyin2Hanzi import DefaultDagParams, dag
from pypinyin import lazy_pinyin
from itertools import product
import random

initials = ['', 'b','p','m','f','d','t','n','l','g','k','h','j','q','x','zh','ch','sh','r','z','c','s','y','w']

finals = [
    'a','o','e','i','u','ü',
    'ai','ei','ui','ao','ou','iu','ie','üe',
    'er','an','en','in','un','ün',
    'ang','eng','ing','ong',
    'ia','ua','uo','uai','iao','ian','uan','uang','iang','iong','üe'
]

similar_initials = [
    ['b', 'p', 'm', 'f'],
    ['d', 't', 'n', 'l'],
    ['g', 'k', 'h'],
    ['j', 'q', 'x'],
    ['zh', 'ch', 'sh', 'r'],
    ['z', 'c', 's'],
    ['y', 'w', ''],
]

similar_finals = [
    ['a', 'ia', 'ua', 'ao', 'iao', 'uao'],
    ['an', 'ian', 'uan'],
    ['ang', 'iang', 'uang'],
    ['ei', 'ui', 'uai'],
    ['ou', 'iu', 'iou'],
    ['e', 'ie', 'üe'],
    ['i', 'in', 'ing', 'ian', 'iang', 'iong'],
    ['u', 'un', 'ong', 'uang', 'ua', 'uo'],
    ['ü', 'ün', 'iong', 'üan'],
    ['ai', 'ei'],
    ['an', 'en', 'in', 'un', 'ün'],
    ['ang', 'eng', 'ing', 'ong'],
]

def get_char_by_pinyin(pinyin):
    dag_params = DefaultDagParams()
    results = dag(dag_params, [pinyin], path_num=1)
    if results:
        return results[0].path[0]
    return None

def split_initial_final(pinyin):
    initials = ['zh','ch','sh','b','p','m','f','d','t','n','l','g','k','h','j','q','x','r','z','c','s','y','w']
    for ini in initials:
        if pinyin.startswith(ini):
            return ini, pinyin[len(ini):]
    return '', pinyin

def in_same_group(char, groups):
    for group in groups:
        if char in group:
            return group
    return None

def similar_initial(ini1, ini2):
    return in_same_group(ini1, similar_initials) == in_same_group(ini2, similar_initials)

def similar_final(fin1, fin2):
    return in_same_group(fin1, similar_finals) == in_same_group(fin2, similar_finals)

def get_similar_pinyins(pinyin):
    ini1, fin1 = split_initial_final(pinyin)
    results = []
    for ini in initials:
        for fin in finals:
            candidate = ini + fin
            if candidate == pinyin:
                continue
            if similar_initial(ini1, ini) and similar_final(fin1, fin):
                results.append(candidate)
    return results

def get_similar_chars(ch):
    result = []
    py = lazy_pinyin(ch)[0]
    similar_pys = get_similar_pinyins(py)
    for similar_py in similar_pys:
        similar_char = get_char_by_pinyin(similar_py)
        if similar_char:
            result.append(similar_char)
    return result

def get_similar_texts(text):
    similar_chars_list = []
    for ch in text:
        similar_chars_list.append(get_similar_chars(ch))
    all_combinations = product(*similar_chars_list)
    return [''.join(combo) for combo in all_combinations]

def generate_adversarial_texts(text, N):
    texts = get_similar_texts(text)
    length = len(texts)
    if N <= length:
        return random.sample(texts, N)
    else:
        result = random.sample(texts, N)
        extra_count = N - length
        result += random.choices(texts, k=extra_count)
        random.shuffle(result)
        return result
# 测试
text = '小特小特'
adversarial_texts = generate_adversarial_texts(text, 100)
print("输入文本:", text)
print("相近字集合:", '\n'.join(adversarial_texts))
print("相近字集合数量:", len(adversarial_texts))