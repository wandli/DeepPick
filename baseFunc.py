import json
import csv
import pandas as pd


# 读取有信息user Dic
def readInfoUserDic():
    urDic = {}
    urNumDic = {}
    with open("yelp_dataset/info_user_dic.txt", "r", encoding='utf-8') as fu:
        while True:
            line = fu.readline()
            if not line:
                break
            l = line.strip('\n').split(' ')
            urDic[l[0]] = int(l[1])
            urNumDic[int(l[1])] = l[0]
    return urDic, urNumDic


def get_special_userlist():
    head = 'yelp_dataset/predict/'
    with open(head + 'shs.txt') as f:
        shs = f.read().strip().split('\n')
    with open(head + 'shsord.txt') as f:
        shs_ord = f.read().strip().split('\n')
    with open(head + 'prord.txt') as f:
        pr_ord = f.read().strip().split('\n')
    with open(head + 'pr.txt') as f:
        pr = f.read().strip().split('\n')
        pr2 = []
        for u in pr:
            pr2.append(u.split(' ')[0])
    return shs, shs_ord, pr2, pr_ord


def get_all_clean():
    bz2ur = {}
    ur2bz = {}
    cat = {}
    csvFile = open("yelp_dataset/businessNum.txt", "r")
    reader = csv.reader(csvFile)
    for r in reader:
        cat[r[1]] = r[0]
        bz2ur[r[1]] = []
        urs = r[3].split("|")
        for u in urs:
            if u:
                bz2ur[r[1]].append(u)
                if u not in ur2bz:
                    ur2bz[u] = []
                ur2bz[u].append(r[1])
    csvFile.close()
    return bz2ur, ur2bz, cat


# 重写清洗后的数据
def Clean():
    csvFile = open("/yelp_dataset/business.txt", "r")
    reader = csv.reader(csvFile)
    bzDic, bzNumDic = readBzizDic()
    urDic, urNumDic = readInfoUserDic()
    wt = open("yelp_dataset/businessNum.txt", "w", newline='')
    writer = csv.writer(wt)
    for r in reader:
        us = r[3].split('|')
        u = ""
        for x in us:
            u += str(urDic[x]) + "|"
        w = [r[0], bzDic[r[1]], r[2], u]
        writer.writerow(w)
    wt.close()
    csvFile.close()


# 读取business Dic
def readBzizDic():
    bzDic = {}
    bzNumDic = {}
    with open("yelp_dataset/business_dic.txt", "r", encoding='utf-8') as fu:
        while True:
            line = fu.readline()
            if not line:
                break
            l = line.strip('\n').split('\t')
            bzDic[l[0]] = int(l[1])
            bzNumDic[int(l[1])] = l[0]
    return bzDic, bzNumDic


# 读取entropy
def get_entropy(year):
    entropy = {}
    with open("yelp_dataset/entropy/business_entropy" + year + ".txt", "r", encoding='utf-8') as fu:
        while True:
            line = fu.readline()
            if not line:
                break
            l = line.strip('\n').split('\t')
            entropy[int(l[0])] = float(l[1])
    return entropy


# 已有reviewDic的情况下进行读取
# return bz->[user]
def readReviewDic(filename):
    reviewDic = {}
    with open("yelp_dataset/" + filename, "r", encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            b = line.strip('\n').split('\t')
            reviewDic[int(b[0])] = []
            reviewer = b[1].split(' ')
            for i in range(1, len(reviewer)):
                reviewDic[int(b[0])].append(int(reviewer[i]))
    return reviewDic


# 读取business->users[]
# year为空时读取全部信息
def get_bz_ur(year):
    with open("yelp_dataset/business_user_edge" + year+ ".json", "r", encoding='utf-8') as fp:
        bz2ur = json.load(fp)
    return bz2ur


# 读取brokerage文件
def get_brokerage(year):
    brokerage = {}
    with open("yelp_dataset/brokerage/bz_all_brokerage"+year+".txt", "r", encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            l = line.split('\t')
            brokerage[int(l[0])] = float(l[-1])
    return brokerage


# 读取city dic, city number stored as integer
def get_citydic():
    city = {}
    with open("yelp_dataset/category/citydic.txt") as fp:
        x = fp.read().split('\n')
        for c in x:
            y = c.split(',')
            city[int(y[0])] = y[1]
    return city


# 读取cat dic, categories number stored as str
def get_catdic():
    with open("yelp_dataset/catdic.json", "r") as fp:
        cat = json.load(fp)
    return cat


# 读取homogenity
def get_homo():
    homo = {}
    with open("yelp_dataset/homogeneity.txt", "r") as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            l = line.split(',')
            if float(l[3]) < 0:
                b = 1
            homo[l[1]] = float(l[3])
    return homo


# 读取infomation graph APG编号
def read_infoAPG():
    APG = []
    with open("yelp_dataset/info_SHS_APG.txt", "r", encoding='utf-8') as fs:
        r = fs.read()
        x = r.split('\n')
        for i in x:
            if len(i) < 10:
                break
            j = i.split(' ')
            if int(j[-1]) in APG:
                continue
            APG.append(int(j[-1]))
    return APG


# 读取interest graph APG编号
def read_interAPG(year):
    APG = []
    with open("yelp_dataset/shs" + year + ".txt", "r", encoding='utf-8') as fs:
        r = fs.read()
        x = r.split('\n')
        for i in x:
            if len(i) < 10:
                break
            j = i.split(' ')
            APG.append(int(j[-1]))
    return APG


# 读取HISid
def readHISid(urNumDic):
    HIS = []
    with open("yelp_dataset/HIS_structural_hole.txt", "r", encoding='utf-8') as fs:
        r = fs.read()
        x = r.split('\n')
        for i in x:
            if len(i) < 1:
                break
            HIS.append(urNumDic[int(i)])
    return HIS


# 由边信息构造用户图，由于输入改为info_user_graph无需再检验是否是有信息用户
# def readGraph(infoId):
def readGraph():
    graph = {}
    with open("yelp_dataset/info_user_graph.txt", "r", encoding='utf-8') as fs:
        while True:
            line = fs.readline()
            if not line:
                break
            l = line.strip('\n').split(' ')
            u1 = int(l[0])
            u2 = int(l[1])
        # if u1 in infoId and u2 in infoId:
            if u1 not in graph:
                graph[u1] = set()
            if u2 not in graph:
                graph[u2] = set()
            graph[u1].add(u2)
            graph[u2].add(u1)
    return graph


# 在info图每个id上+1
def add_info_user():
    f = open("yelp_dataset/info_user_graph_1start.txt", "w", encoding='utf-8')
    with open("yelp_dataset/info_user_graph.txt", "r", encoding='utf-8') as fs:
        while True:
            line = fs.readline()
            if not line:
                break
            l = line.strip('\n').split(' ')
            u1 = int(l[0])
            u2 = int(l[1])
            f.write(str(u1 + 1) + ' ' + str(u2 + 1) + '\n')
    f.close()


# 从带权的review + tip graph打印不带权review graph
def uw_review(year):
    fp = open("yelp_dataset/year[04-18]_integrate_edge/Uinte_edge"+year+".txt", "w", encoding='utf-8')
    with open("yelp_dataset/year[04-18]_integrate_edge/"+year+"_integrate_edge.txt", "r", encoding='utf-8') as fu:
        while True:
            line = fu.readline()
            if not line:
                break
            l = line.strip('\n').split(' ')
            fp.write(l[0] + ' ' + l[1] + '\n')
    fp.close()


def readUserRT():
    rows = []
    with open("yelp_dataset/review.json", "r", encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            tmp = json.loads(line)
            rows.append([tmp['user_id'], tmp['business_id'], tmp['date'], tmp['text']])

    with open("yelp_dataset/tip.json", "r", encoding='utf-8') as ft:
        while True:
            line = ft.readline()
            if not line:
                break
            tmp = json.loads(line)
            rows.append([tmp['user_id'], tmp['business_id'], tmp['date'], tmp['text']])

    headers = ['userid', 'businessid', 'date', 'text']
    with open('yelp_dataset/text.csv', 'w', encoding='utf-8', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)


# 把地点数据整合
def wash_loc():
    header = 'yelp_dataset/new_user/burt_shs/esize_efficiency/'
    head = ['user_id', 'homo_avg', 'homo_mid', 'homo_min', 'homo_max', 'homo_small_q',
            'homo_large_q',  'en_avg', 'en_mid', 'en_min','en_max', 'en_small_q',
            'en_large_q', 'bro_avg', 'bro_mid',  'bro_min', 'bro_max', 'bro_small_q',
            'bro_large_q']
    f1 = pd.read_csv('yelp_dataset/AllUserBz.txt', sep="\t", header=None).drop(19, axis=1)
    f3 = pd.read_csv('yelp_dataset/predict/loc.csv')
    f3 = f3[f3['user_id'] > 1]
    f1.columns=head
    f = pd.concat([f1, f3], axis=0, ignore_index=True, sort=False).drop_duplicates(subset=['user_id'])
    f.to_csv('yelp_dataset/predict/loc2.csv', index=False)


# get k-core users
def get_kcore():
    with open('yelp_dataset/predict/k-shell/shell.txt', 'r', encoding='utf-8') as f:
        kcore = f.read().split('\n')
        kcore = [int(i) for i in kcore]
    with open('yelp_dataset/predict/k-shell/shellord.txt', 'r', encoding='utf-8') as f:
        kcoreord = f.read().split('\n')
        kcoreord = [int(i.split(' ')[0]) for i in kcoreord]
    return kcore, kcoreord


def get_esize():
    header = 'yelp_dataset/new_user/burt_shs/hierachy_constraint/'
    s = pd.read_csv(header + 'SHS_constraint.txt', sep=' ')
    o = pd.read_csv(header + 'ord_constraint.txt', sep=' ')
    return list(s['id']), list(o['id'])


def get_elite():
    header = 'yelp_dataset/new_user/elite/'
    with open(header + 'elite.txt') as f:
        s = f.read().strip().split('\n')
    with open(header + 'elite_ord.txt') as f:
        o = f.read().strip().split('\n')
    return s, o


def get_const():
    header = 'yelp_dataset/new_user/const/'
    with open(header + 'const.txt') as f:
        s = f.read().strip().split('\n')
    with open(header + 'constord.txt') as f:
        o = f.read().strip().split('\n')
    return s, o


def get_betweenness():
    header = 'yelp_dataset/new_user/Betweeness/'
    s = pd.read_csv(header + 'betw.txt', sep=' ', header=None)
    o = pd.read_csv(header + 'betword.txt', sep=' ', header=None)
    return list(s[0]), list(o[0])


if __name__ == '__main__':
    # uw_review('18')
    # readUserRT()
    wash_loc()
