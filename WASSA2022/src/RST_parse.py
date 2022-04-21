# -*- coding: UTF-8 -*-
import ast
import os


def alignment_check(listRST):
    file = open('../data/test_essays.txt', 'r')
    lines = file.readlines()
    file.close()
    unaligned = []
    for i in range(len(listRST)):
        sent = ' '.join(listRST[i].keys())
        if sent.replace(" ", "")[:20] != lines[i].replace(" ", "").replace("\n", "")[:20]:
            print('The following sentences are not aligned!')
            print(' '.join(listRST[i].keys()))
            print(lines[i])
            unaligned.append(lines[i])
            break
        else:
            continue
    return unaligned


def get_RST_dict(directory='../RST-output'):
    listRST = []
    for filename in sorted(os.listdir(directory), key=lambda f: int(f.rsplit(os.path.extsep, 1)[0].rsplit('out', 1)[-1])):
        with open(os.path.join(directory, filename), 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
        dict = {}
        for item in ast.literal_eval(last_line):
            dict.update({item[0]: item[4]})

        listRST.append(dict)
    return listRST


def extract_nuclei(directory='../RST-output', check_alignment=False):
    listRST = get_RST_dict(directory=directory)

    extracted = []
    if check_alignment:
        if not alignment_check(listRST):
            for rst_dict in listRST:
                temp = []
                for key in rst_dict.keys():
                    if rst_dict[key] == 0:
                        temp.append(key.strip())
                extracted.append(' '.join(temp))

    else:
        for rst_dict in listRST:
            temp = []
            for key in rst_dict.keys():
                if rst_dict[key] == 0:
                    temp.append(key.strip())
            extracted.append(' '.join(temp))
    return extracted
